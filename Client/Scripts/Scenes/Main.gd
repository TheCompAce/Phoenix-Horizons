extends Node

#Imports
const NetworkClient = preload("res://Scripts/Network/websockets_client.gd")
const Packet = preload("res://Scripts/Network/packet.gd")
const Chatbox = preload("res://Scenes/UI/Chat/Chatbox.tscn")
const Actor = preload("res://Scenes/Actors/Actor.tscn")

onready var _login_screen = get_node("Login")
onready var _network_client = NetworkClient.new()
var state: FuncRef
var _chatbox = null
var _username: String
var _player_actor = null
var _actors: Dictionary = {}

func _ready():
	_network_client.connect("connected", self, "_handle_client_connected")
	_network_client.connect("disconnected", self, "_handle_client_disconnected")
	_network_client.connect("error", self, "_handle_network_error")
	_network_client.connect("data", self, "_handle_network_data")
	add_child(_network_client)
	_network_client.connect_to_server("127.0.0.1", 6438)
	
	_login_screen.connect("login", self, "_handle_login_button")
	_login_screen.connect("register", self, "_handle_register_button")
	
	state = null

func _handle_client_connected():
	print("Client connected to server!")

func _handle_client_disconnected(was_clean: bool):
	print("Disconnected %s" % ["cleanly" if was_clean else "unexpectedly"])
	get_tree().quit()
	
func _handle_network_data(data: String):
	print("Received server data: ", data)
	var action_payloads: Array = Packet.json_to_action_payloads(data)
	var p: Packet = Packet.new(action_payloads[1], action_payloads[2])
	
	# Pass the packet to our current state
	state.call_func(p)
	
func _handle_network_error():
	print("There was an error.")

func _unhandled_input(event):
	if _player_actor and event.is_action_released("click"):
		var target = _player_actor.body.get_global_mouse_position()
		_player_actor._player_target = target
		var p: Packet = Packet.new("Action.Target", [target.x, target.y])
		_network_client.send_packet(p)

func LOGIN(p):
	print(p.action)
	match p.action:
		"Action.Ok":
			print("OK")
			_enter_game()
		"Action.Deny":
			print("Deny")
			var reason: String = p.payloads[0]
			OS.alert(reason)

func _handle_login_button(username: String, password: String):
	_username = username
	state = funcref(self, "LOGIN")
	var p: Packet = Packet.new("Action.Login", [username, password])
	_network_client.send_packet(p)
	
func REGISTER(p):
	print(p.action)
	match p.action:
		"Action.Ok":
			OS.alert("Registration succesful.")
		"Action.Deny":
			var reason: String = p.payloads[0]
			OS.alert(reason)

func _handle_register_button(username: String, password: String, email: String, avatar_id: int):
	state = funcref(self, "REGISTER")
	var p: Packet = Packet.new("Action.Register", [username, password, email, avatar_id])
	_network_client.send_packet(p)

func _enter_game():
	print("_enter_game")
	state = funcref(self, "PLAY")
	
	# Remove Login screen
	remove_child(_login_screen)
	
	# Start Chatbox
	_chatbox = Chatbox.instance()
	_chatbox.connect("message_sent", self, "send_chat")
	add_child(_chatbox)
	
func _update_models(model_data: Dictionary):
	print("Received model data: " + JSON.print(model_data))
	var model_id: int = model_data["id"]
	var func_name: String = "_update_" + model_data["model_type"].to_lower()
	print(func_name)
	var f: FuncRef = funcref(self, func_name)
	f.call_func(model_id, model_data)

func PLAY(p):
	match p.action:
		"Action.Chat":
			var sender: String = p.payloads[0]
			var message: String = p.payloads[1]
			_chatbox.add_message(sender, message)
		"Action.ModelDelta":
			var model_data: Dictionary = p.payloads[0]
			_update_models(model_data)
	
func send_chat(text: String):
	var p: Packet = Packet.new("Action.Chat", [_username, text])
	_network_client.send_packet(p)
	_chatbox.add_message(_username, text)

# Dynamic functions called from "_update_models"
func _update_actor(model_id: int, model_data: Dictionary):
	if model_id in _actors:
		_actors[model_id].update(model_data)
	else:
		var new_actor
		if not _player_actor:
			var actor = Actor.instance()
			if actor:
				print(actor)
				_player_actor = actor.init(model_data)
				_player_actor.is_player = true
				new_actor = _player_actor
			else:
				print("Error")
		else:
			new_actor = Actor.instance().init(model_data)
			
		_actors[model_id] = new_actor
		add_child(new_actor)
