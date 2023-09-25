extends Node

const Packet = preload("res://Scripts/Network/packet.gd")
const html_key = preload("res://Scripts/Tools/key.gd")
signal connected
signal data
signal disconnected
signal error
# Our WebSocketCLient tnstance
var _client = WebSocketClient.new()

var crypto = Crypto.new()
var key = CryptoKey.new()
var client_private_key_data = ""
var client_public_key_data = ""
var server_public_key_data = ""

func _ready():
	_client.connect("connection_closed", self, "_closed")
	_client.connect("connection_error", self, "_error")
	_client.connect("connection_established", self, "_connected")
	_client.connect("data_received", self, "_on_data")
	
	# Bad but works
	_client.verify_ssl = false

	
	
func connect_to_server(hostname: String, port: int) -> void:
	# Connects to the server or emits an error signal.
	# If connected, emits a connect signal.
	var websocket_url = "wss://%s:%d" % [hostname, port]
	var err = _client.connect_to_url(websocket_url)	
	if err:
		print( "Unable to connect." )
		set_process(false)
		emit_signal("error")

func get_server_public_key() -> String:
	return server_public_key_data

func send_packet(packet: Packet) -> void:
	var packet_data = packet.to_string()

	_send_string(packet_data)

func _closed (was_clean = false):
	print( "Closed, clean: ", was_clean)
	set_process(false)
	emit_signal("disconnected", was_clean)
	
func _connected(proto = ""):
	print("Connected with protocol: ", proto)
	emit_signal("connected")


func _on_data():
	var data: String = _client.get_peer(1).get_packet().get_string_from_utf8()
	# print(data)
	if client_private_key_data != "":
		data = crypto.decrypt(key, data.to_utf8())
	
	# print( "Got data from server: ", data)	
	emit_signal("data", data)
		
	
	
func _process(delta):
	_client.poll()

func _send_string(string: String) -> void:
	_client.get_peer(1).put_packet((string.to_utf8()))
	print("Sent string ", string)

func _error():
	print("There was an Error!")
