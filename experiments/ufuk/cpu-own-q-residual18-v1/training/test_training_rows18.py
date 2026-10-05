import numpy as np
from training_cli18_v2 import _json_groups
def test_both_native_json_list_and_numpy_training_rows_keep_same_exact_data():
 row=dict(x=list(range(18)),prior_logit=.25,target=-.5)
 actual=_json_groups({"a":[row]})
 array_row={**row,"x":np.asarray(row["x"],dtype=np.float64)}
 assert actual==_json_groups({"a":[array_row]})
 assert actual["a"][0]["x"]==list(range(18))
