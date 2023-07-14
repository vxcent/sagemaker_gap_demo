import json
import os
import _jsonnet
import os
import torch
from seq2struct.commands.infer import Inferer
from seq2struct.datasets.spider import SpiderItem
from seq2struct.utils import registry
from seq2struct.datasets.spider_lib.preprocess.get_tables import dump_db_json_schema
from seq2struct.datasets.spider import load_tables_from_schema_dict

# Reads in configurations of the model, the directory of the model, and the specific checkpoint of the model
# Returns:
#   - inferer: data structure enabling model loading
#   - model_dir: directory PATh corresponds to the bart model
#   - checkpoint_step: helps specify selection of file
def load_inferer():
    exp_config = json.loads(
        _jsonnet.evaluate_file(
            "experiments/spider-configs/gap-run.jsonnet"))
    model_config_path = exp_config["model_config"]
    model_config_args = exp_config.get("model_config_args")
    infer_config = json.loads(
        _jsonnet.evaluate_file(
            model_config_path, 
            tla_codes={'args': json.dumps(model_config_args)}))
    infer_config["model"]["encoder_preproc"]["db_path"] = "data/sqlite_files/"
    model_dir = exp_config["logdir"] + "/bs=12,lr=1.0e-04,bert_lr=1.0e-05,end_lr=0e0,att=1"
    checkpoint_step = exp_config["eval_steps"][0]
    return Inferer(infer_config), model_dir, checkpoint_step

# Loads a model object based on weights and metadata
# Params:
#   - inferer
#   - model_dir
#   - checkpoint_step
# Returns: 
#   - Inference model within the inferer class
def load_model(inferer, model_dir, checkpoint_step):
    return inferer.load_model(model_dir, checkpoint_step)

# Loads schema registry from SQLite files stored locally
# Param:
#   - db_id: Database target of interest
# Returns:
#   - dataset: collection of schema information 
def schema_dataset_dump(db_id):
    my_schema = dump_db_json_schema("data/sqlite_files/{db_id}/{db_id}.sqlite".format(db_id=db_id), db_id)
    schema, eval_foreign_key_maps = load_tables_from_schema_dict(my_schema)
    dataset = registry.construct('dataset_infer', {
        "name": "spider", "schemas": schema, "eval_foreign_key_maps": eval_foreign_key_maps, 
        "db_path": "data/sqlite_files/"
    })
    return dataset

# Preprocessing of model based on specific database schema of interest
# Returns:
#  - model: model with preprocessed schema
def pre_processing(model, dataset):
    for _, schema in dataset.schemas.items():
        model.preproc.enc_preproc._preprocess_schema(schema)
    return model

# Inference of model based on question and schema
# Params:
#   - question: natural language question from user
#   - spider_schema: schema of database of interest
#   - model: model object
#   - inferer: inferer object
# Returns:
#   - output: inferred SQL code
def infer(question, spider_schema, model, inferer):
    data_item = SpiderItem(
            text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
            code=None,
            schema=spider_schema,
            orig_schema=spider_schema.orig,
            orig={"question": question}
        )
    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
    preproc_data = enc_input, None
    with torch.no_grad():
        output = inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)
    return output[0]["inferred_code"]

