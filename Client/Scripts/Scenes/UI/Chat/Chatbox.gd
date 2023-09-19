extends Control

onready var chat_log = get_node("CanvasLayer/VBoxContainer/RichTextLabel")
onready var input_field = get_node("CanvasLayer/VBoxContainer/HBoxContainer/LineEdit")
onready var send_button = get_node("CanvasLayer/VBoxContainer/HBoxContainer/Button")

signal message_sent(message)

func _ready():
	input_field.connect("text_entered", self, "text_entered")
	send_button.connect("pressed", self, "send_button_pressed")
	
func send_button_pressed():
	text_entered(input_field.text)
	
func _input(event: InputEvent):
	if event is InputEventKey and event.pressed:
		match event.scancode:
			KEY_ENTER:
				input_field.grab_focus()
			KEY_ESCAPE:
				input_field.release_focus()
	
func text_entered(text: String):
	if len(text) > 0:
		input_field.text = ""
		emit_signal("message_sent", text)

func add_message(username: String, text: String):
	var line = "[" + username + "]" + text + "\n"
	chat_log.bbcode_text += line
