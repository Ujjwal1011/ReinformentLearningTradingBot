import socketio
import hmac
import hashlib
import json
import time

# CoinDCX WebSocket endpoint
socketEndpoint = 'https://stream.coindcx.com'

# --- Replace with your actual API credentials ---
key = "5a2310f46c09e5df47bc3468ac4742f81ff1b22885f5b222"
secret = "8d0fe394b2e0d43f3ba75220c2cd071d2f3c8fee904434351b316e069e47ab14"
channel_name = 'B-ETH_USDT_1m-futures'

# Create the SocketIO client
sio = socketio.Client(logger=True, engineio_logger=True)

# Track last open_time to avoid duplicates
last_candle_open_time = None

@sio.event
def connect():
    print("✅ Connected to CoinDCX WebSocket")

    # Prepare authentication signature
    body = {"channelName": channel_name}
    json_body = json.dumps(body, separators=(',', ':'))
    signature = hmac.new(secret.encode(), json_body.encode(), hashlib.sha256).hexdigest()

    # Join the channel
    payload = {
        "channelName": channel_name,
        "authSignature": signature,
        "apiKey": key
    }
    sio.emit('join', payload)
    print(f"--> Subscribed to {channel_name}")

@sio.event
def disconnect():
    print("❌ Disconnected from CoinDCX WebSocket")

@sio.on('candlestick')
def on_candlestick(response):
    global last_candle_open_time

    try:
        # Parse the nested JSON string
        parsed = json.loads(response['data'])
        candle = parsed['data'][0]  # We assume only one candle per update
        open_time = candle['open_time']

        # Only show if it's a new candle (new minute)
        if open_time != last_candle_open_time:
            last_candle_open_time = open_time
            print("\n--- 🕯️ 1-Min Candlestick ---")
            print(json.dumps(candle, indent=2))

    except Exception as e:
        print(f"⚠️ Error processing candlestick data: {e}")

# Run the connection
if __name__ == '__main__':
    try:
        print("Attempting to connect...")
        sio.connect(socketEndpoint, transports=['websocket'])
        sio.wait()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if sio.connected:
            sio.disconnect()
