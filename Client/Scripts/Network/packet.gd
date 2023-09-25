extends Object

var crypto = Crypto.new()
var key = CryptoKey.new()

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
	var encrypted_data = crypto.encrypt(key, data.to_utf8())

	# Convert the PoolByteArray to Base64 encoded String
	var base64_encoded: String = base64_encode(encrypted_data)
	
	return base64_encoded

	

static func json_to_action_payloads(base64_str: String, key) -> Array:
	var id: String
	var action: String
	var payloads: Array = []
	var obj_dict: Dictionary
	var crypto = Crypto.new()
	
	if key == "":
		obj_dict = JSON.parse(base64_str).result
	else:
		var decoded_data = base64_decode(base64_str)  # Decode from Base64
		
		var decrypted_data = crypto.decrypt(key, decoded_data)
		obj_dict = JSON.parse(decrypted_data.get_string_from_utf8()).result

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


const BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

static func base64_encode(data: PoolByteArray) -> String:
	var output = ""
	var char_array = []
	var padding = 0
	
	for i in range(0, data.size(), 3):
		var byte1 = data[i]
		var byte2 = 0
		var byte3 = 0

		if i + 1 < data.size():
			byte2 = data[i + 1]
		if i + 2 < data.size():
			byte3 = data[i + 2]
		
		var num = (byte1 << 16) + (byte2 << 8) + byte3
		
		padding = 3 - ((data.size() - i) if (data.size() - i) < 3 else 3)
		
		for j in range(4):
			if j < 4 - padding:
				char_array.append(BASE64_CHARS[(num >> ((3 - j) * 6)) & 0x3F])
			else:
				char_array.append("=")
	
	output = "".join(char_array)
	return output


const BASE64_CHARS_INV = {
	"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9,
	"K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19,
	"U":20, "V":21, "W":22, "X":23, "Y":24, "Z":25, "a":26, "b":27, "c":28, "d":29,
	"e":30, "f":31, "g":32, "h":33, "i":34, "j":35, "k":36, "l":37, "m":38, "n":39,
	"o":40, "p":41, "q":42, "r":43, "s":44, "t":45, "u":46, "v":47, "w":48, "x":49,
	"y":50, "z":51, "0":52, "1":53, "2":54, "3":55, "4":56, "5":57, "6":58, "7":59,
	"8":60, "9":61, "+":62, "/":63, "=":64
}

static func base64_decode(data: String) -> PoolByteArray:
	var output = PoolByteArray()
	var char_array = data.split("")
	var padding = 0
	
	for i in range(0, char_array.size() - 1, 4):
		var b1 = BASE64_CHARS_INV[char_array[i]]
		var b2 = BASE64_CHARS_INV[char_array[i + 1]]
		var b3 = BASE64_CHARS_INV[char_array[i + 2]]
		var b4 = BASE64_CHARS_INV[char_array[i + 3]]
		
		var num = (b1 << 18) + (b2 << 12) + (b3 << 6) + b4
		
		output.append((num >> 16) & 0xFF)
		output.append((num >> 8) & 0xFF)
		output.append(num & 0xFF)
		
	return output

