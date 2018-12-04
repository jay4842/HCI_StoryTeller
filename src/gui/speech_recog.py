import speech_recognition as sr
import time

class Speech_Recogn:
    '''
        # Constructor method
        # Initialize wordBank, recognizer, & microphone
        # Hey Dr. R! If you're reading this, then we hope it puts a detectable smile on your face :)
    '''
    def __init__(self):
        #wordbank of acceptable string types
        self.wordBank = ["capture", "exit"]

        #set up reconizer and microphone objects
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    '''
        # Method: will set up microphone to transcribe speech

        # Return:
            # Success: a bool indicating whether or not API request was successful
            # Error: 'None' if no error occurred otherwise a string with error message
                if API did not work or speech unrecognizable
            # Transcription: 'None' if speech could not be recognized otherwise string
                containing transcribed text
    '''
    def recognize_speech_from_mic(self):
        if not isinstance(self.recognizer, sr.Recognizer):
            raise TypeError("'recognizer' must be 'Recognizer' instance")
        if not isinstance(self.microphone, sr.Microphone):
            raise TypeError("'microphone' must be 'Microphone' instance")

        #adjust recognizer sensitivity to ambient noise and record audio from 
        #microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        #set up the response object
        response ={
            "Success": True,
            "Error": None,
            "Transcription": None
        }

        try:
            response["Transcription"] = self.recognizer.recognize_google(audio)
        except sr.RequestError:
            #API was unreachable or unresponsive
            response["Success"] = False
            response["Error"] = "API unavailable"
        except sr.UnknownValueError:
            # speech was unintelligible
            response["Error"] = "Unable to recognize speech"

        return response

    def speech_runner(self):
        #loop for three seconds
        t_end = time.time() + (60 * 3)
        heard_something = False
        print("Listening...")
        while(time.time() < t_end):
            self.speech_cmd = self.recognize_speech_from_mic()
            if self.speech_cmd["Transcription"]:
                heard_something = True
                break
            if not self.speech_cmd["Success"]:
                break
            #print("I didn't catch that. Please re-re-re-repeat")
        
        #if error, stop the program
        if self.speech_cmd["Error"]:
            #print("Error: {}".format(self.speech_cmd["Error"]))
            return None
        # if it heard something
        if(heard_something):
            #show user transcription
            #print("You said: {}". format(self.speech_cmd["Transcription"]))
            for word in self.wordBank:
                if(self.speech_cmd["Transcription"].lower() == word):
                    #print(word + '!')
                    return word

        return None # if nothing was caught

'''if __name__ == "__main__":
    s = Speech_Recogn()
    s.speech_runner()'''
