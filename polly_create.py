import json
import os

import boto3


def text_to_speech_polly(text, output_filepath, voice_id="Mizuki"):
    """
    Convert text to speech using Amazon Polly and save to a file with the specified format.

    Parameters:
    - text: Text to be converted to speech.
    - output_filepath: Filepath where the speech will be saved.
    - voice_id: The ID of the voice to use (default is 'Mizuki', a Japanese voice).
    """
    # Create a Polly client
    polly = boto3.client(
        "polly",
        region_name="ap-northeast-1",
    )

    try:
        # Request speech synthesis
        response = polly.synthesize_speech(
            Text=text, OutputFormat="mp3", VoiceId=voice_id
        )

        # Save the audio stream to a file
        with open(output_filepath, "wb") as audio_file:
            audio_file.write(response["AudioStream"].read())

        print(f"Generated {output_filepath}")
    except Exception as e:
        print(f"Failed to generate {output_filepath}: {str(e)}")


def create_speech_from_json(json_filepath, output_folder):
    try:
        with open(json_filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

            for key, sub_data in data.items():
                for sub_key, text_data in sub_data.items():
                    # Check if reading_info is available, otherwise use guide_info
                    text = text_data.get("reading_info") or text_data.get(
                        "guide_info", ""
                    )
                    # Ensure `text` is a string
                    if not isinstance(text, str):
                        text = str(text)

                    output_filepath = os.path.join(
                        output_folder, f"wm{int(key):05}_{sub_key}.mp3"
                    )

                    # Check if the file already exists
                    if os.path.exists(output_filepath):
                        print(
                            f"File {output_filepath} already exists. Skipping..."
                        )
                        continue

                    text_to_speech_polly(text, output_filepath)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


create_speech_from_json("./案内データ2020.json", "message")
