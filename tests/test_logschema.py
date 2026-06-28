"""Round-trip tests for the logging schema (logschema.py), the data contract.

Guards that StepWriter actually writes Parquet that reads back with the declared
row count and dtypes (the buffered writer infers array types then relies on the
schema to cast float64->float32, int64->int32, list<double>->list<float32>), that
incomplete records are rejected, and that the run manifest round-trips via JSON.
Skips cleanly if pyarrow is not installed (it is a lazy/optional dependency).
"""

import json
import os
import tempfile

import pytest

from logschema import (
    STEP_COLUMNS,
    RunManifest,
    StepWriter,
    new_step_record,
    write_manifest,
)

pa = pytest.importorskip("pyarrow")


def test_new_step_record_has_every_column():
    assert set(new_step_record()) == set(STEP_COLUMNS)


def test_step_writer_roundtrip_preserves_rows_and_types():
    import pyarrow.parquet as pq

    path = os.path.join(tempfile.mkdtemp(), "steps.parquet")
    writer = StepWriter(path, batch_size=2)  # 5 rows spans multiple flush batches
    for i in range(5):
        rec = new_step_record()
        rec["run_id"] = "r1"
        rec["step_idx"] = i
        rec["obs"] = [float(i), 0.0, 1.0]
        rec["action"] = [0.1, 0.2, 0.0, 0.0, 0.0]
        rec["reward"] = float(i)
        rec["species_id"] = 2
        writer.append(rec)
    writer.close()

    table = pq.read_table(path)
    assert table.num_rows == 5
    assert table.column("step_idx").to_pylist() == [0, 1, 2, 3, 4]
    assert table.schema.field("reward").type == pa.float32()
    assert table.schema.field("species_id").type == pa.int32()
    assert table.schema.field("obs").type.value_type == pa.float32()


def test_step_writer_rejects_incomplete_record():
    path = os.path.join(tempfile.mkdtemp(), "steps.parquet")
    writer = StepWriter(path)
    rec = new_step_record()
    del rec["reward"]
    with pytest.raises(KeyError):
        writer.append(rec)


def test_manifest_roundtrips_through_json():
    manifest = RunManifest(
        run_id="r1", experiment="A", ladder_level="L1_discretization",
        ladder_params={"delta": 0.06}, obs_spec_hash="abc", action_spec_hash="def",
        channel_mask={"vision": True, "smell": False, "intero": True},
        seeds={"world": 1}, world_params={"L": 1.0}, agent_config={},
        episode_horizon=30, n_episodes=10,
    )
    back = json.load(open(write_manifest(tempfile.mkdtemp(), manifest)))
    assert back["run_id"] == "r1"
    assert back["ladder_params"]["delta"] == 0.06
    assert back["channel_mask"]["smell"] is False
