"""Sanity check that every module imports cleanly."""

def test_package_imports():
    import mol
    from mol import MoLModel, TopKMoLModel, TriRouter
    from mol import SKIP, EXEC, REPEAT
    from mol import recovery, data, eval, router, model, topk_model
    assert (SKIP, EXEC, REPEAT) == (0, 1, 2)
    assert hasattr(mol, "__version__")


def test_recovery_api():
    from mol.recovery import (
        add_lora, set_lora, set_lora_trainable, lora_parameters,
        kd_loss, block_influence, pretrain_to_static,
        train_stage_a, train_topk_stage_a, train_stage_b,
        apply_route,
    )
    # All public functions are callables
    assert all(callable(f) for f in (add_lora, set_lora, kd_loss,
                                       block_influence, train_stage_a,
                                       train_topk_stage_a, train_stage_b,
                                       apply_route))


def test_eval_api():
    from mol.eval import wikitext_ppl, mc_accuracy, eight_task_accuracy
    assert all(callable(f) for f in (wikitext_ppl, mc_accuracy, eight_task_accuracy))
