import numpy as np
import json
import os
import utils
import torch.utils.data as torch_data
from wikisql_gendata import SQLExample
from collections import defaultdict
from typing import List

stats = defaultdict(int)

class InputFeature(object):
    def __init__(self,
                 question,
                 table_id,
                 tokens,
                 word_to_char_start,
                 word_to_subword,
                 subword_to_word,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.question = question
        self.table_id = table_id
        self.tokens = tokens
        self.word_to_char_start = word_to_char_start
        self.word_to_subword = word_to_subword
        self.subword_to_word = subword_to_word
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.columns = None
        self.agg = None
        self.select = None
        self.where_num = None
        self.where = None
        self.op = None
        self.value_start = None
        self.value_end = None

    def output_SQ(self, agg = None, sel = None, conditions = None, return_str=True):
        agg_ops = ['NA', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        cond_ops = ['=', '>', '<', 'OP']

        if agg is None and sel is None and conditions is None:
            sel = np.argmax(self.select)
            agg = self.agg[sel]
            conditions = []
            for i in range(len(self.where)):
                if self.where[i] == 0:
                    continue
                conditions.append((i, self.op[i], self.value_start[i], self.value_end[i]))

        agg_text = agg_ops[agg]
        select_text = self.columns[sel]
        cond_texts = []
        for wc, op, vs, ve in conditions:
            column_text = self.columns[wc]
            op_text = cond_ops[op]
            word_start, word_end = self.subword_to_word[wc][vs], self.subword_to_word[wc][ve]
            char_start = self.word_to_char_start[word_start]
            char_end = len(self.question) if word_end + 1 >= len(self.word_to_char_start) else self.word_to_char_start[word_end + 1]
            value_span_text = self.question[char_start:char_end]
            cond_texts.append(column_text + op_text + value_span_text.rstrip())

        if return_str:
            sq = agg_text + ", " + select_text + ", " + " AND ".join(cond_texts)
        else:
            sq = (agg_text, select_text, set(cond_texts))

        return sq

class HydraFeaturizer(object):
    def __init__(self, config):
        self.config = config
        self.tokenizer = utils.create_tokenizer(config)
        self.colType2token = {
            "string": "[unused1]",
            "real": "[unused2]"}

    def get_input_feature(self, example: SQLExample, config):
        max_total_length = int(config["max_total_length"])

        input_feature = InputFeature(
            example.question,
            example.table_id,
            [],
            example.word_to_char_start,
            [],
            [],
            [],
            [],
            []
        )

        for column, col_type, _ in example.column_meta:
            # get query tokens
            tokens = []
            word_to_subword = []
            subword_to_word = []
            for i, query_token in enumerate(example.tokens):
                if self.config["base_class"] == "roberta":
                    sub_tokens = self.tokenizer.tokenize(query_token, add_prefix_space=True)
                else:
                    sub_tokens = self.tokenizer.tokenize(query_token)
                cur_pos = len(tokens)
                if len(sub_tokens) > 0:
                    word_to_subword += [(cur_pos, cur_pos + len(sub_tokens))]
                    tokens.extend(sub_tokens)
                    subword_to_word.extend([i] * len(sub_tokens))

            if self.config["base_class"] == "roberta":
                tokenize_result = self.tokenizer.encode_plus(
                    col_type + " " + column,
                    tokens,
                    padding="max_length",
                    max_length=max_total_length,
                    truncation=True,
                    add_prefix_space=True
                )
            else:
                tokenize_result = self.tokenizer.encode_plus(
                    col_type + " " + column,
                    tokens,
                    padding="max_length",
                    max_length=max_total_length,
                    truncation_strategy="longest_first",
                    truncation=True,
                )

            input_ids = tokenize_result["input_ids"]
            input_mask = tokenize_result["attention_mask"]

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            column_token_length = 0
            if self.config["base_class"] == "roberta":
                for i, token_id in enumerate(input_ids):
                    if token_id == self.tokenizer.sep_token_id:
                        column_token_length = i + 2
                        break
                segment_ids = [0] * max_total_length
                for i in range(column_token_length, max_total_length):
                    if input_mask[i] == 0:
                        break
                    segment_ids[i] = 1
            else:
                for i, token_id in enumerate(input_ids):
                    if token_id == self.tokenizer.sep_token_id:
                        column_token_length = i + 1
                        break
                segment_ids = tokenize_result["token_type_ids"]

            subword_to_word = [0] * column_token_length + subword_to_word
            word_to_subword = [(pos[0]+column_token_length, pos[1]+column_token_length) for pos in word_to_subword]

            assert len(input_ids) == max_total_length
            assert len(input_mask) == max_total_length
            assert len(segment_ids) == max_total_length

            input_feature.tokens.append(tokens)
            input_feature.word_to_subword.append(word_to_subword)
            input_feature.subword_to_word.append(subword_to_word)
            input_feature.input_ids.append(input_ids)
            input_feature.input_mask.append(input_mask)
            input_feature.segment_ids.append(segment_ids)

        return input_feature

    def fill_label_feature(self, example: SQLExample, input_feature: InputFeature, config):
        max_total_length = int(config["max_total_length"])

        columns = [c[0] for c in example.column_meta]
        col_num = len(columns)
        input_feature.columns = columns

        input_feature.agg = [0] * col_num
        input_feature.agg[example.select] = example.agg
        input_feature.where_num = [len(example.conditions)] * col_num

        input_feature.select = [0] * len(columns)
        input_feature.select[example.select] = 1

        input_feature.where = [0] * len(columns)
        input_feature.op = [0] * len(columns)
        input_feature.value_start = [0] * len(columns)
        input_feature.value_end = [0] * len(columns)

        for colidx, op, _ in example.conditions:
            input_feature.where[colidx] = 1
            input_feature.op[colidx] = op
        for colidx, column_meta in enumerate(example.column_meta):
            if column_meta[-1] == None:
                continue
            se = example.value_start_end[column_meta[-1]]
            try:
                s = input_feature.word_to_subword[colidx][se[0]][0]
                input_feature.value_start[colidx] = s
                e = input_feature.word_to_subword[colidx][se[1]-1][1]-1
                input_feature.value_end[colidx] = e
                assert s < max_total_length and input_feature.input_mask[colidx][s] == 1
                assert e < max_total_length and input_feature.input_mask[colidx][e] == 1

            except:
                print("value span is out of range")
                return False

        # feature_sq = input_feature.output_SQ(return_str=False)
        # example_sq = example.output_SQ(return_str=False)
        # if feature_sq != example_sq:
        #     print(example.qid, feature_sq, example_sq)
        return True

    def load_data(self, data_paths, config, include_label=False):
        model_inputs = {k: [] for k in ["input_ids", "input_mask", "segment_ids"]}
        if include_label:
            for k in ["agg", "select", "where_num", "where", "op", "value_start", "value_end"]:
                model_inputs[k] = []

        pos = []
        input_features = []
        for data_path in data_paths.split("|"):
            cnt = 0
            for line in open(data_path, encoding="utf8"):
                example = SQLExample.load_from_json(line)
                if not example.valid and include_label == True:
                    continue

                input_feature = self.get_input_feature(example, config)
                if include_label:
                    success = self.fill_label_feature(example, input_feature, config)
                    if not success:
                        continue

                # sq = input_feature.output_SQ()
                input_features.append(input_feature)

                cur_start = len(model_inputs["input_ids"])
                cur_sample_num = len(input_feature.input_ids)
                pos.append((cur_start, cur_start + cur_sample_num))

                model_inputs["input_ids"].extend(input_feature.input_ids)
                model_inputs["input_mask"].extend(input_feature.input_mask)
                model_inputs["segment_ids"].extend(input_feature.segment_ids)
                if include_label:
                    model_inputs["agg"].extend(input_feature.agg)
                    model_inputs["select"].extend(input_feature.select)
                    model_inputs["where_num"].extend(input_feature.where_num)
                    model_inputs["where"].extend(input_feature.where)
                    model_inputs["op"].extend(input_feature.op)
                    model_inputs["value_start"].extend(input_feature.value_start)
                    model_inputs["value_end"].extend(input_feature.value_end)

                cnt += 1
                if cnt % 5000 == 0:
                    print(cnt)

                if "DEBUG" in config and cnt > 100:
                    break

        for k in model_inputs:
            model_inputs[k] = np.array(model_inputs[k], dtype=np.int64)

        return input_features, model_inputs, pos

class SQLDataset(torch_data.Dataset):
    def __init__(self, data_paths, config, featurizer, include_label=False):
        self.config = config
        self.featurizer = featurizer
        self.input_features, self.model_inputs, self.pos = self.featurizer.load_data(data_paths, config, include_label)

        print("{0} loaded. Data shapes:".format(data_paths))
        for k, v in self.model_inputs.items():
            print(k, v.shape)

    def __len__(self):
        return self.model_inputs["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.model_inputs.items()}


if __name__ == "__main__":
    vocab = "vocab/baseTrue.txt"
    config = {}
    for line in open("conf/wikisql.conf", encoding="utf8"):
        if line.strip() == "" or line[0] == "#":
             continue
        fields = line.strip().split("\t")
        config[fields[0]] = fields[1]
    # config["DEBUG"] = 1

    featurizer = HydraFeaturizer(config)
    train_data = SQLDataset(config["train_data_path"], config, featurizer, True)
    train_data_loader = torch_data.DataLoader(train_data, batch_size=128, shuffle=True, pin_memory=True)
    for batch_id, batch in enumerate(train_data_loader):
        print(batch_id, {k: v.shape for k, v in batch.items()})

    for k, v in stats.items():
        print(k, v)