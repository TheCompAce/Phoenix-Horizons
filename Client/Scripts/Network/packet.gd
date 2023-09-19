extends Object

var id: String
var action: String
var payloads: Array

const uuid_util = preload('res://Scripts/Tools/uuid.gd')

func _init(_action: String, _payloads: Array):
	randomize()
	id = uuid_util.v4()
	action = _action
	payloads = _payloads
	
	
func to_string() -> String:
	var dict: Dictionary = {"id": id, "a": str(action)}
	for i in range(len(payloads)):
		dict["p%d" % i] = payloads[i]
		
	var data: String = JSON.print(dict)
	return data
	
static func json_to_action_payloads(json_str: String) -> Array:
	var id: String
	var action: String
	var payloads: Array = []
	print(json_str)
	var obj_dict: Dictionary = JSON.parse(json_str).result
	
	for key in obj_dict.keys():
		var value = obj_dict[key]
		if key == "id":
			id = value
		elif key == "a":
			action = value
		elif key[0] == "p":
			var index: int = key.split_floats("p", true)[1]
			payloads.insert(index, value)
			
	return [id, action, payloads]
