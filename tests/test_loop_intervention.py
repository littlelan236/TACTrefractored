import json
import pytest
from pydantic import ValidationError
from tact.config import Agent, agent_config
from tact.rollout.agent import ActionCycleDetector, DurableAgent, action_signature
from tact.research.trajectories import TrajectoryReader


def message(command, id='random'):
    return {'role':'assistant','content':'', 'tool_calls':[{'id':id,'function':{'name':'bash','arguments':json.dumps({'command':command})}}], 'extra':{'actions':[command]}}

@pytest.mark.parametrize('period', range(1,6))
def test_two_cycles(period):
    detector=ActionCycleDetector()
    for i in range(period*2):
        result=detector.observe(message(str(i%period)),i)
        assert (result is not None)==(i==period*2-1)
    assert result['period']==period
    assert result['matched_steps']==list(range(period*2))


def test_reset_and_parameters():
    d=ActionCycleDetector()
    assert d.observe(message('A'),0) is None
    assert d.observe(message('A'),1)['period']==1
    assert d.observe(message('A'),2) is None
    assert d.observe(message('B'),3) is None
    d.observe({'role':'assistant'},4)
    assert d.observe(message('B'),5) is None
    assert action_signature(message('A','one'))==action_signature(message('A','two'))
    assert action_signature(message("echo 'a  b'"))!=action_signature(message("echo 'a b'"))
    other=message('A');other['tool_calls'][0]['function']['name']='other'
    assert action_signature(other)!=action_signature(message('A'))

@pytest.mark.parametrize('cfg',[{'max_period':0},{'max_period':6},{'min_repeats':1},{'prompt':''},{'prompt':'   '}])
def test_invalid_config(cfg):
    with pytest.raises(ValidationError):Agent(loop_intervention=cfg)

class Model:
    def __init__(self):self.seen=[];self.i=0
    def query(self,messages):
        self.seen.append([dict(m) for m in messages]);self.i+=1
        return message('A',str(self.i))
    def format_message(self,**kwargs):return kwargs
    def format_observation_messages(self,message,outputs,variables):
        return [{'role':'tool','content':'done','tool_call_id':message['tool_calls'][0]['id']}]
    def serialize(self):return {}
    def get_template_vars(self):return {}
class Env:
    def execute(self,action):return {'output':'done'}
    def serialize(self):return {}
    def get_template_vars(self):return {}

@pytest.mark.parametrize('enabled',[False,True])
def test_agent_context_and_persistence(tmp_path,enabled):
    model=Model();agent=DurableAgent(model,Env(),loop_intervention={'enabled':enabled},system_template='system',instance_template='task',step_limit=3,output_path=tmp_path/'t.traj.json')
    result=agent.run('task')
    assert result['exit_status']=='LimitsExceeded'
    assert len(model.seen)==3
    assert [m['role'] for m in model.seen[2]]==(['system','user','assistant','tool','assistant','tool','user'] if enabled else ['system','user','assistant','tool','assistant','tool'])
    assert len(agent.loop_events)==(1 if enabled else 0)
    reader=TrajectoryReader(tmp_path/'t.traj.json')
    if enabled:
        event=agent.loop_events[0]
        assert event['matched_steps']==[0,1]
        assert reader.prepared_prefix(7)[-1]['content']==agent.loop_settings.prompt
        assert reader.prepared_prefix(7)[-1]['role']=='user'
        assert agent.loop_settings.prompt not in reader.prepared_prefix(4)[0]['content']
        assert json.loads((tmp_path/'loop-interventions.jsonl').read_text())['period']==1
        assert reader.data['info']['loop_intervention']['count']==1
    else:assert not (tmp_path/'loop-interventions.jsonl').exists()

def test_detector_isolation_and_custom_settings():
    a=ActionCycleDetector(1,3);b=ActionCycleDetector()
    assert a.observe(message('A'),0) is None
    assert a.observe(message('A'),1) is None
    assert b.observe(message('A'),0) is None
    assert a.observe(message('A'),2)['repeats']==3


def test_submission_skips_injection():
    class SubmittedModel(Model):
        def format_observation_messages(self,*args):return [{'role':'exit','extra':{'exit_status':'Submitted'}}]
    a=DurableAgent(SubmittedModel(),Env(),loop_intervention={'enabled':True},system_template='system',instance_template='task')
    a.loop_detector.observe(message('A'),0)
    a.execute_actions(message('A'))
    assert a.messages[-1]['role']=='exit'
    assert not a.loop_events



def test_config_reaches_agent_and_hash_changes():
    from tact.config import resolve_config
    a=resolve_config();b=resolve_config(overrides=['agent.loop_intervention.enabled=true','agent.loop_intervention.prompt=Try a different approach.'])
    assert a['config_hash']!=b['config_hash']
    agent=DurableAgent(Model(),Env(),**agent_config(b))
    assert agent.loop_settings.enabled
    assert agent.loop_settings.prompt=='Try a different approach.'


def test_repeated_reminders_and_new_actions(tmp_path):
    class SequenceModel(Model):
        def query(self,messages):
            self.seen.append([dict(m) for m in messages])
            cmd=['A','A','A','A','B','C'][self.i];self.i+=1
            return message(cmd,str(self.i))
    model=SequenceModel()
    a=DurableAgent(model,Env(),loop_intervention={'enabled':True,'prompt':'Explore alternatives.'},system_template='base',instance_template='task',step_limit=6,output_path=tmp_path/'t.json')
    a.run('task')
    assert [e['trigger_step'] for e in a.loop_events]==[1,3]
    assert model.seen[2][-1]['content']=='Explore alternatives.'
    assert model.seen[4][-1]['content']=='Explore alternatives.'
    assert all(m[0]['content']=='base' for m in model.seen)
    assert len((tmp_path/'loop-interventions.jsonl').read_text().splitlines())==2
    a.save(tmp_path/'t.json')
    assert len((tmp_path/'loop-interventions.jsonl').read_text().splitlines())==2


def test_exception_breaks_pending_cycle():
    class FailedEnv(Env):
        def execute(self,action):raise RuntimeError('failed action')
    a=DurableAgent(Model(),FailedEnv(),loop_intervention={'enabled':True},system_template='base',instance_template='task')
    a.loop_detector.observe(message('A'),0)
    with pytest.raises(RuntimeError):a.step()
    assert not a.loop_detector.history
    assert not a.loop_events


def test_disabled_matches_upstream(tmp_path):
    from minisweagent.agents.default import DefaultAgent
    a=DefaultAgent(Model(),Env(),system_template='base',instance_template='task',step_limit=3)
    b=DurableAgent(Model(),Env(),system_template='base',instance_template='task',step_limit=3)
    assert a.run('task')==b.run('task')
    assert a.messages==b.messages
    assert a.n_calls==b.n_calls


def test_multiple_actions_order_and_json_format():
    a=message('A');b=message('B');a['tool_calls']+=b['tool_calls']
    reverse={**a,'tool_calls':list(reversed(a['tool_calls']))}
    assert action_signature(a)!=action_signature(reverse)
    x=message('A');x['tool_calls'][0]['function']['arguments']={'command':'A'}
    assert action_signature(x)==action_signature(message('A'))
