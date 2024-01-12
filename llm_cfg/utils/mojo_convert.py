import os
from mxeval.data import read_problems, stream_jsonl, write_jsonl, get_metadata

def get_datafile(dataset="mbxp", language="python"):
  metadata, datadir = get_metadata(dataset, metadata_type="problem")
  if language.lower() not in metadata:
    raise ValueError(f"Language {language} not found in metadata file")
  datafile = metadata[language.lower()]
  return os.path.join(datadir, datafile)

dataset = "multi-humaneval"
problem_file=get_datafile(dataset, 'python')
problems = read_problems(problem_file)

for id, problem in problems.items():
#   print(problem['prompt'])
    # print(problem["test"])
    print(problem['prompt'] + '\n' + problem["canonical_solution"])
    # print(f"check({problem['entry_point']})")
    break
