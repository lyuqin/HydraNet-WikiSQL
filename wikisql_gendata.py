import json
import os
import string
import unicodedata
import utils

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\n" or c == "\r":
        return True
    cat = unicodedata.category(c)
    if cat == "Zs":
        return True
    return False

def is_punctuation(c):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(c)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(c)
  if cat.startswith("P") or cat.startswith("S"):
    return True
  return False

def basic_tokenize(doc):
    doc_tokens = []
    char_to_word = []
    word_to_char_start = []
    prev_is_whitespace = True
    prev_is_punc = False
    prev_is_num = False
    for pos, c in enumerate(doc):
        if is_whitespace(c):
            prev_is_whitespace = True
            prev_is_punc = False
        else:
            if prev_is_whitespace or is_punctuation(c) or prev_is_punc or (prev_is_num and not str(c).isnumeric()):
                doc_tokens.append(c)
                word_to_char_start.append(pos)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
            prev_is_punc = is_punctuation(c)
            prev_is_num = str(c).isnumeric()
        char_to_word.append(len(doc_tokens) - 1)

    return doc_tokens, char_to_word, word_to_char_start

class SQLExample(object):
    def __init__(self,
                 qid,
                 question,
                 table_id,
                 column_meta,
                 agg=None,
                 select=None,
                 conditions=None,
                 tokens=None,
                 char_to_word=None,
                 word_to_char_start=None,
                 value_start_end=None,
                 valid=True):
        self.qid = qid
        self.question = question
        self.table_id = table_id
        self.column_meta = column_meta
        self.agg = agg
        self.select = select
        self.conditions = conditions
        self.valid = valid
        if tokens is None:
            self.tokens, self.char_to_word, self.word_to_char_start = basic_tokenize(question)
            self.value_start_end = {}
            if conditions is not None and len(conditions) > 0:
                cur_start = None
                for cond in conditions:
                    value = cond[-1]
                    value_tokens, _, _ = basic_tokenize(value)
                    val_len = len(value_tokens)
                    for i in range(len(self.tokens)):
                        if " ".join(self.tokens[i:i+val_len]).lower() != " ".join(value_tokens).lower():
                            continue
                        s = self.word_to_char_start[i]
                        e = len(question) if i + val_len >= len(self.word_to_char_start) else self.word_to_char_start[i + val_len]
                        recovered_answer_text = question[s:e].strip()
                        if value.lower() == recovered_answer_text.lower():
                            cur_start = i
                            break

                    if cur_start is None:
                        self.valid = False
                        print([value, value_tokens, question, self.tokens])
                        # for c in question:
                        #     print((c, ord(c), unicodedata.category(c)))
                        # raise Exception()
                    else:
                        self.value_start_end[value] = (cur_start, cur_start + val_len)
        else:
            self.tokens, self.char_to_word, self.word_to_char_start, self.value_start_end = tokens, char_to_word, word_to_char_start, value_start_end

    @staticmethod
    def load_from_json(s):
        d = json.loads(s)
        keys = ["qid", "question", "table_id", "column_meta", "agg", "select", "conditions", "tokens", "char_to_word", "word_to_char_start", "value_start_end", "valid"]

        return SQLExample(*[d[k] for k in keys])

    def dump_to_json(self):
        d = {}
        d["qid"] = self.qid
        d["question"] = self.question
        d["table_id"] = self.table_id
        d["column_meta"] = self.column_meta
        d["agg"] = self.agg
        d["select"] = self.select
        d["conditions"] = self.conditions
        d["tokens"] = self.tokens
        d["char_to_word"] = self.char_to_word
        d["word_to_char_start"] = self.word_to_char_start
        d["value_start_end"] = self.value_start_end
        d["valid"] = self.valid

        return json.dumps(d)

    def output_SQ(self, return_str=True):
        agg_ops = ['NA', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        cond_ops = ['=', '>', '<', 'OP']

        agg_text = agg_ops[self.agg]
        select_text = self.column_meta[self.select][0]
        cond_texts = []
        for wc, op, value_text in self.conditions:
            column_text = self.column_meta[wc][0]
            op_text = cond_ops[op]
            cond_texts.append(column_text + op_text + value_text)

        if return_str:
            sq = agg_text + ", " + select_text + ", " + " AND ".join(cond_texts)
        else:
            sq = (agg_text, select_text, set(cond_texts))
        return sq

def get_schema(tables):
    schema, headers, colTypes, naturalMap = {}, {}, {}, {}
    for table in tables:
        values = [set() for _ in range(len(table["header"]))]
        for row in table["rows"]:
            for i, value in enumerate(row):
                values[i].add(str(value).lower())
        columns = {column: values[i] for i, column in enumerate(table["header"])}

        trans = {"text": "string", "real": "real"}
        colTypes[table["id"]] = {col:trans[ty] for ty, col in zip(table["types"], table["header"])}
        schema[table["id"]] = columns
        naturalMap[table["id"]] = {col: col for col in columns}
        headers[table["id"]] = table["header"]

    return schema, headers, colTypes, naturalMap



if __name__ == "__main__":
    data_path = os.path.join("WikiSQL", "data")
    for phase in ["train", "dev", "test"]:
        src_file = os.path.join(data_path, phase + ".jsonl")
        schema_file = os.path.join(data_path, phase + ".tables.jsonl")
        output_file = os.path.join("data", "wiki" + phase + ".jsonl")
        schema, headers, colTypes, naturalMap = get_schema(utils.read_jsonl(schema_file))

        cnt = 0
        print("processing {0}...".format(src_file))
        with open(output_file, "w", encoding="utf8") as f:
            for raw_sample in utils.read_jsonl(src_file):
                table_id = raw_sample["table_id"]
                sql = raw_sample["sql"]

                cur_schema = schema[table_id]
                header = headers[table_id]
                cond_col_values = {header[cond[0]]: str(cond[2]) for cond in sql["conds"]}
                column_meta = []
                for col in header:
                    if col in cond_col_values:
                        column_meta.append((col, colTypes[table_id][col], cond_col_values[col]))
                    else:
                        detected_val = None
                        # for cond_col_val in cond_col_values.values():
                        #     if cond_col_val.lower() in cur_schema[col]:
                        #         detected_val = cond_col_val
                        #         break
                        column_meta.append((col, colTypes[table_id][col], detected_val))

                example = SQLExample(
                    cnt,
                    raw_sample["question"],
                    table_id,
                    column_meta,
                    sql["agg"],
                    int(sql["sel"]),
                    [(int(cond[0]), cond[1], str(cond[2])) for cond in sql["conds"]])

                f.write(example.dump_to_json() + "\n")
                cnt += 1

                # if cnt % 1000 == 0 and cnt > 0:
                #     print(cnt)