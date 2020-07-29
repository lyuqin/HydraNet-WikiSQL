import os
import numpy as np
import utils
from modeling.base_model import BaseModel
from modeling.model_factory import create_model
from featurizer import InputFeature, HydraFeaturizer, SQLDataset

class HydraEvaluator():
    def __init__(self, output_path, config, hydra_featurizer: HydraFeaturizer, model:BaseModel, note=""):
        self.config = config
        self.model = model
        self.eval_history_file = os.path.join(output_path, "eval.log")
        self.bad_case_dir = os.path.join(output_path, "bad_cases")
        if "DEBUG" not in config:
            os.mkdir(self.bad_case_dir)
            with open(self.eval_history_file, "w", encoding="utf8") as f:
                f.write(note.rstrip() + "\n")

        self.eval_data = {}
        for eval_path in config["dev_data_path"].split("|") + config["test_data_path"].split("|"):
            eval_data = SQLDataset(eval_path, config, hydra_featurizer, True)
            self.eval_data[os.path.basename(eval_path)] = eval_data

            print("Eval Data file {0} loaded, sample num = {1}".format(eval_path, len(eval_data)))

    def _eval_imp(self, eval_data: SQLDataset, get_sq=True):
        items = ["overall", "agg", "sel", "wn", "wc", "op", "val"]
        acc = {k:0.0 for k in items}
        sq = []
        cnt = 0
        model_outputs = self.model.dataset_inference(eval_data)
        for input_feature, model_output in zip(eval_data.input_features, model_outputs):
            cur_acc = {k:1 for k in acc if k != "overall"}

            select_label = np.argmax(input_feature.select)
            agg_label = input_feature.agg[select_label]
            wn_label = input_feature.where_num[0]
            wc_label = [i for i, w in enumerate(input_feature.where) if w == 1]

            agg, select, where, conditions = self.model.parse_output(input_feature, model_output, wc_label)
            if agg != agg_label:
                cur_acc["agg"] = 0
            if select != select_label:
                cur_acc["sel"] = 0
            if len(where) != wn_label:
                cur_acc["wn"] = 0
            if set(where) != set(wc_label):
                cur_acc["wc"] = 0

            for w in wc_label:
                _, op, vs, ve = conditions[w]
                if op != input_feature.op[w]:
                    cur_acc["op"] = 0

                if vs != input_feature.value_start[w] or ve != input_feature.value_end[w]:
                    cur_acc["val"] = 0

            for k in cur_acc:
                acc[k] += cur_acc[k]

            all_correct = 0 if 0 in cur_acc.values() else 1
            acc["overall"] += all_correct

            if ("DEBUG" in self.config or get_sq) and not all_correct:
                try:
                    true_sq = input_feature.output_SQ()
                    pred_sq = input_feature.output_SQ(agg=agg, sel=select, conditions=[conditions[w] for w in where])
                    task_cor_text = "".join([str(cur_acc[k]) for k in items if k in cur_acc])
                    sq.append([str(cnt), input_feature.question, "|".join([task_cor_text, pred_sq, true_sq])])
                except:
                    pass
            cnt += 1

        result_str = []
        for item in items:
            result_str.append(item + ":{0:.1f}".format(acc[item] * 100.0 / cnt))

        result_str = ", ".join(result_str)

        return result_str, sq

    def eval(self, epochs):
        print(self.bad_case_dir)
        for eval_file in self.eval_data:
            result_str, sq = self._eval_imp(self.eval_data[eval_file])
            print(eval_file + ": " + result_str)

            if "DEBUG" in self.config:
                for text in sq:
                    print(text[0] + ":" + text[1] + "\t" + text[2])
            else:
                with open(self.eval_history_file, "a+", encoding="utf8") as f:
                    f.write("[{0}, epoch {1}] ".format(eval_file, epochs) + result_str + "\n")

                bad_case_file = os.path.join(self.bad_case_dir,
                                           "{0}_epoch_{1}.log".format(eval_file, epochs))
                with open(bad_case_file, "w", encoding="utf8") as f:
                    for text in sq:
                        f.write(text[0] + ":" + text[1] + "\t" + text[2] + "\n")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    config = utils.read_conf(os.path.join("conf", "wikisql.conf"))
    config["DEBUG"] = 1
    config["num_train_steps"] = 1000
    config["num_warmup_steps"] = 100

    featurizer = HydraFeaturizer(config)
    model = create_model(config, is_train=True, num_gpu=1)
    evaluator = HydraEvaluator("output", config, featurizer, model, "debug evaluator")
    evaluator.eval(0)