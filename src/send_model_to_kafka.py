# Copyright 2025 Konstantinos Karavitis
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from confluent_kafka import Producer
import pathlib

# --- Config ---
BROKER = "localhost:9092"           # Change to your broker URL
TOPIC  = "ml-model"
ZIP_FILE = pathlib.Path("data/pricing_saved_model.zip")

# --- Load model as bytes ---
with open(ZIP_FILE, "rb") as f:
  payload = f.read()

# --- Set up Kafka producer ---
producer = Producer({"bootstrap.servers": BROKER})

# --- Send it (async) ---
producer.produce(
  TOPIC,
  value=payload,
  key="pricing-model-v1"           # optional key (for partitioning)
)

producer.flush()
print(f"✅ Sent {ZIP_FILE.name} to Kafka topic '{TOPIC}'")