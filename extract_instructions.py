import json

LLM_gen_instr_types = ["ABE6", "D_3", "D_4"]

filename = "25k_with_ablation_noE5E6E2S1/train.json"
instructs = list()

with open(filename, 'r') as f:
    for i, data in enumerate(f):
        content = json.loads(data)
        if content["instr_type"] in LLM_gen_instr_types:
            last_sentence_sep = content["instruction"].rindex("\n\n")
            instr = content["instruction"][last_sentence_sep:].strip()
            if instr[0] == '"':
                instr = instr.strip('"')
            instructs.append(instr)

#print(len(instructs))
with open(filename[:-5] + "_extracted.txt", 'w') as f:
    for i in instructs:
        f.write(i + "\n\n")
