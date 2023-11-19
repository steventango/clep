import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast, CLIPVisionModelWithProjection

hostName = "localhost"
serverPort = 32829


map_digit_to_token = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

map_digit_to_input_ids = {
    digit: tokenizer(f"{map_digit_to_token[digit]}", padding=True).input_ids for digit in range(10)
}

text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")


import torch

map_digit_to_embedding = {
    digit: text_model(
        input_ids=torch.tensor(map_digit_to_input_ids[digit]).unsqueeze(0),
        attention_mask=torch.ones(1, 3),
    ).text_embeds
    for digit in range(10)
}

C = 4
H, W = 168, 400
L = 400
S = max(H, W)

from wavelet_spectrogram import cwt_spectrogram

map_digit_to_input_ids = {
    digit: tokenizer(f"{map_digit_to_token[digit]}", padding=True).input_ids for digit in range(10)
}


def preprocess(inputs):
    B = inputs["eeg"].shape[0]
    samples = np.zeros((B, C, S, S))
    eeg = np.array(inputs["eeg"])
    for b in range(B):
        for i in range(C):
            sample = eeg[b, i]
            power, *_ = cwt_spectrogram(sample, 200, nNotes=24, detrend=True, normalize=True)
            samples[b, i, : min(H, S), : min(W, S)] = power.squeeze()
    inputs["pixel_values"] = samples
    return inputs


embeddings = torch.concat([map_digit_to_embedding[digit] for digit in range(10)])
embeddings = embeddings.to("cuda")


model = CLIPVisionModelWithProjection.from_pretrained("clep_trainer_our_data/checkpoint-169/")
model.eval()
model.to("cuda")


class MyServer(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        obj = json.loads(body)
        eeg = np.asarray(obj["eeg"])
        if eeg.ndim == 2:
            eeg = np.expand_dims(eeg, 0)
        inputs = preprocess({"eeg": eeg})
        with torch.no_grad():
            outputs = model(pixel_values=torch.tensor(inputs["pixel_values"]).to("cuda"))
        probs = (embeddings @ outputs.image_embeds.T).softmax(dim=0)
        probs = probs.cpu().detach().numpy().squeeze()
        response = {map_digit_to_token[digit]: float(probs[digit]) for digit in range(10)}
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))


def main():
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")


if __name__ == "__main__":
    main()
