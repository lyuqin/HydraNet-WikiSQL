import os
import json
import pickle
import utils
from modeling.model_factory import create_model
from featurizer import HydraFeaturizer, SQLDataset
from wikisql_lib.dbengine import DBEngine

def print_metric(label_file, pred_file):
    sp = [(json.loads(ls)["sql"], json.loads(lp)["query"]) for ls, lp in zip(open(label_file), open(pred_file))]

    sel_acc = sum(p["sel"] == s["sel"] for s, p in sp) / len(sp)
    agg_acc = sum(p["agg"] == s["agg"] for s, p in sp) / len(sp)
    wcn_acc = sum(len(p["conds"]) == len(s["conds"]) for s, p in sp) / len(sp)

    def wcc_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [c[0] for c in a] == [c[0] for c in b]

    def wco_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [c[1] for c in a] == [c[1] for c in b]

    def wcv_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [str(c[2]).lower() for c in a] == [str(c[2]).lower() for c in b]

    wcc_acc = sum(wcc_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)
    wco_acc = sum(wco_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)
    wcv_acc = sum(wcv_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)

    print('sel_acc: {}\nagg_acc: {}\nwcn_acc: {}\nwcc_acc: {}\nwco_acc: {}\nwcv_acc: {}\n' \
          .format(sel_acc, agg_acc, wcn_acc, wcc_acc, wco_acc, wcv_acc))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # in_file = "data/wikidev.jsonl"
    # db_file = "data/wikisql/dev.db"
    # out_file = "wikisql_master/dev_out.jsonl"
    # label_file = "wikisql_master/data/dev.jsonl"
    # model_out_file = "wikisql_master/dev_model_out.pkl"

    in_file = "data/wikitest.jsonl"
    db_file = "data/wikisql/test.db"
    out_file = "wikisql_master/test_out.jsonl"
    label_file = "wikisql_master/data/test.jsonl"
    model_out_file = "wikisql_master/test_model_out.pkl"

    # Old Best
    # model_path = "output/20200117_230155"
    # epoch = 4

    # My Best
    # model_path = "output/20200207_103903"
    # epoch = 3

    # All Best
    model_path = "output/20200207_105347"
    epoch = 4

    # BERT
    # model_path = "output/20200226_151300"
    # epoch = 4

    engine = DBEngine(db_file)
    config = utils.read_conf(os.path.join(model_path, "model.conf"))
    # config["DEBUG"] = 1
    featurizer = HydraFeaturizer(config)
    pred_data = SQLDataset(in_file, config, featurizer, False)
    print("num of samples: {0}".format(len(pred_data.input_features)))

    model = create_model(config, is_train=False)
    model.load(model_path, epoch)

    if "DEBUG" in config:
        model_out_file = model_out_file + ".partial"
    model_outputs = model.dataset_inference(pred_data)
    pickle.dump(model_outputs, open(model_out_file, "wb"))
    # model_outputs = pickle.load(open(model_out_file, "rb"))

    print("===HydraNet===")
    pred_sqls = model.predict_SQL(pred_data, model_outputs=model_outputs)
    with open(out_file, "w") as g:
        for pred_sql in pred_sqls:
            # print(pred_sql)
            result = {"query": {}}
            result["query"]["agg"] = int(pred_sql[0])
            result["query"]["sel"] = int(pred_sql[1])
            result["query"]["conds"] = [(int(cond[0]), int(cond[1]), str(cond[2])) for cond in pred_sql[2]]
            g.write(json.dumps(result) + "\n")
    print_metric(label_file, out_file)

    print("===HydraNet+EG===")
    pred_sqls = model.predict_SQL_with_EG(engine, pred_data, model_outputs=model_outputs)
    with open(out_file + ".eg", "w") as g:
        for pred_sql in pred_sqls:
            # print(pred_sql)
            result = {"query": {}}
            result["query"]["agg"] = int(pred_sql[0])
            result["query"]["sel"] = int(pred_sql[1])
            result["query"]["conds"] = [(int(cond[0]), int(cond[1]), str(cond[2])) for cond in pred_sql[2]]
            g.write(json.dumps(result) + "\n")
    print_metric(label_file, out_file + ".eg")
