import os
import json
import transformers

pretrained_weights = {
    ("bert", "base"): "bert-base-uncased",
    ("bert", "large"): "bert-large-uncased-whole-word-masking",
    ("roberta", "base"): "roberta-base",
    ("roberta", "large"): "roberta-large",
    ("albert", "xlarge"): "albert-xlarge-v2"
}


def read_jsonl(jsonl):
    for line in open(jsonl, encoding="utf8"):
        sample = json.loads(line.rstrip())
        yield sample

def read_conf(conf_path):
    config = {}
    for line in open(conf_path, encoding="utf8"):
        if line.strip() == "" or line[0] == "#":
             continue
        fields = line.strip().split("\t")
        config[fields[0]] = fields[1]
    config["train_data_path"] =  os.path.abspath(config["train_data_path"])
    config["dev_data_path"] =  os.path.abspath(config["dev_data_path"])

    return config

def create_base_model(config):
    weights_name = pretrained_weights[(config["base_class"], config["base_name"])]
    if config["base_class"] == "bert":
        return transformers.BertModel.from_pretrained(weights_name)
    elif config["base_class"] == "roberta":
        return transformers.RobertaModel.from_pretrained(weights_name)
    elif config["base_class"] == "albert":
        return transformers.AlbertModel.from_pretrained(weights_name)
    else:
        raise Exception("base_class {0} not supported".format(config["base_class"]))

def create_tokenizer(config):
    weights_name = pretrained_weights[(config["base_class"], config["base_name"])]
    if config["base_class"] == "bert":
        return transformers.BertTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "roberta":
        return transformers.RobertaTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "albert":
        return transformers.AlbertTokenizer.from_pretrained(weights_name)
    else:
        raise Exception("base_class {0} not supported".format(config["base_class"]))

if __name__ == "__main__":
    qtokens = ['Tell', 'me', 'what', 'the', 'notes', 'are', 'for', 'South', 'Australia']
    column = "string School/Club Team"

    tokenizer = create_tokenizer({"base_class": "roberta", "base_name": "large"})

    qsubtokens = []
    for t in qtokens:
        qsubtokens += tokenizer.tokenize(t, add_prefix_space=True)
    print(qsubtokens)
    result = tokenizer.encode_plus(column, qsubtokens, add_prefix_space=True)
    for k in result:
        print(k, result[k])
    print(tokenizer.convert_ids_to_tokens(result["input_ids"]))



