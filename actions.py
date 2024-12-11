from rasa_sdk import Action
from rasa_sdk.events import SlotSet

class ActionGetTime(Action):
    def name(self):
        return "action_get_time"

    def run(self, dispatcher, tracker, domain):
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        dispatcher.utter_message(text=f"The current time is {current_time}.")
        return []
