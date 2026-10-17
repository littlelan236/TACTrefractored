import multiprocessing
import os
from types import SimpleNamespace

import pytest

from tact.artifacts import read_json, write_json
from tact.rollout.coordination import TaskCoordinator, coordinated_batch


def config(root, workers=1):
    return {'project_root':str(root), 'execution':{'coordination_group':'test-group','rollout_workers':workers},
            'model_descriptor':{'model_fingerprint':'weights'}, 'generation':{'seed':0}, 'agent':{},
            'templates':{}, 'steering':{'strategy':'none'}, 'environment':{}, 'timeouts':{},
            'model':{'max_model_len':262144,'dtype':'bfloat16'},
            'serving':{'backend':'native_vllm','tensor_parallel_size':4,'tool_parser':'qwen3_xml','reasoning_parser':'qwen3'}}


def hold_claim(cfg, run, row, images, pipe):
    coordinator=TaskCoordinator(cfg,run,images)
    claim=coordinator.claim(row)
    pipe.send(claim is not None)
    pipe.recv()
    os._exit(0)


def test_task_lock_survives_contention_and_releases_on_process_death(tmp_path):
    cfg=config(tmp_path);row={'instance_id':'test__one'};images={'test__one':{'image_id':'sha256:test'}}
    parent,child=multiprocessing.Pipe()
    process=multiprocessing.Process(target=hold_claim,args=(cfg,tmp_path/'first',row,images,child))
    process.start()
    try:
        assert parent.recv()
        coordinator=TaskCoordinator(cfg,tmp_path/'second',images)
        assert coordinator.claim(row) is None
        parent.send('exit');process.join(timeout=5)
        assert not process.is_alive()
        claim=coordinator.claim(row);assert claim is not None;claim.close()
    finally:
        if process.is_alive():process.terminate();process.join()


def test_terminal_reuse_keeps_physical_trajectory_and_source_provenance(tmp_path):
    cfg=config(tmp_path);row={'instance_id':'test__one'};images={'test__one':{'image_id':'sha256:test'}}
    source=tmp_path/'first';target=tmp_path/'second'
    result={'instance_id':'test__one','attempt_id':'attempt-000','exit_status':'LimitsExceeded','submission':''}
    traj=source/'tasks/test__one/attempts/attempt-000/test__one.traj.json'
    write_json(traj,{'messages':[]});write_json(traj.parent/'attempt.json',result)
    write_json(source/'tasks/test__one/result.json',result)
    first=TaskCoordinator(cfg,source,images);claim=first.claim(row)
    first.publish(row,result);claim.close()
    second=TaskCoordinator(cfg,target,images);claim=second.claim(row)
    reused=second.reuse(row);claim.close()
    copied=target/'tasks/test__one/attempts'/reused['attempt_id']/'test__one.traj.json'
    assert copied.read_bytes()==traj.read_bytes()
    assert copied.stat().st_ino!=traj.stat().st_ino
    assert reused['reused_from']['run_id']=='first'
    assert read_json(target/'tasks/test__one/result.json')['exit_status']=='LimitsExceeded'
    changed=config(tmp_path);changed['generation']['seed']=9
    with pytest.raises(ValueError,match='incompatible'):
        TaskCoordinator(changed,tmp_path/'third',images)
    mismatch=TaskCoordinator(cfg,tmp_path/'fourth',{'test__one':{'image_id':'different'}})
    with pytest.raises(ValueError,match='mismatch'):mismatch.claim(row)


def test_scheduler_skips_busy_task_and_reuses_when_owner_finishes(tmp_path):
    cfg=config(tmp_path);rows=[{'instance_id':'test__one'},{'instance_id':'test__two'}]
    images={r['instance_id']:{'image_id':'sha256:'+r['instance_id']} for r in rows}
    source=tmp_path/'first';target=tmp_path/'second';owner=TaskCoordinator(cfg,source,images)
    held=owner.claim(rows[0]);executed=[]
    def execute(cfg,run,row,factory):
        executed.append(row['instance_id'])
        result={'instance_id':row['instance_id'],'attempt_id':'attempt-000','exit_status':'Submitted','submission':'patch'}
        write_json(run/'tasks'/row['instance_id']/'attempts/attempt-000'/(row['instance_id']+'.traj.json'),{'messages':[]})
        write_json(run/'tasks'/row['instance_id']/'result.json',result)
        if row['instance_id']=='test__two':
            original={**result,'instance_id':'test__one'}
            write_json(source/'tasks/test__one/attempts/attempt-000/test__one.traj.json',{'messages':[]})
            write_json(source/'tasks/test__one/attempts/attempt-000/attempt.json',original)
            owner.publish(rows[0],original);held.close()
        return result
    try:
        results=coordinated_batch(cfg,target,rows,SimpleNamespace(images=images),execute)
        assert executed==['test__two']
        assert len(results)==2
        assert any(r.get('reused_from') for r in results)
    finally:
        held.close()
