import asyncio
import struct
from typing import Callable, Optional
from bleak import BleakScanner, BleakClient

# Standard GATT UUIDs
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# Custom GATT UUIDs for Wearable
SPO2_MEASUREMENT_UUID = "00002a5f-0000-1000-8000-00805f9b34fb"
TEMP_MEASUREMENT_UUID = "00002a1c-0000-1000-8000-00805f9b34fb"

class BLEReceiver:
    def __init__(self, target_name: str = "HyperSepsis-Band", on_data_received: Optional[Callable] = None):
        self.target_name = target_name
        self.on_data_received = on_data_received
        self.client: Optional[BleakClient] = None
        self.is_connected = False
        self._running = False

    def _parse_hr_measurement(self, data: bytearray) -> int:
        """Parses the standard GATT Heart Rate Measurement value."""
        flags = data[0]
        # Check if 8-bit or 16-bit heart rate format
        if (flags & 0x01) == 0:
            hr = data[1]
        else:
            hr = struct.unpack("<H", data[1:3])[0]
        return hr

    def _notification_handler(self, sender: int, data: bytearray):
        """Processes incoming BLE notifications."""
        # Simple parsing logic
        uuid = str(sender).lower()
        vitals_update = {}
        
        try:
            if "2a37" in uuid: # Heart Rate
                hr = self._parse_hr_measurement(data)
                vitals_update["heart_rate"] = float(hr)
                # Compute mock HRV based on simple interval calculation
                vitals_update["hrv"] = float(max(20.0, min(140.0, 1000.0 / (hr / 60.0) * 0.08))) 
            elif "2a5f" in uuid: # SpO2
                spo2 = struct.unpack("<f", data)[0]
                vitals_update["spo2"] = float(spo2)
            elif "2a1c" in uuid: # Temperature
                temp = struct.unpack("<f", data)[0]
                vitals_update["skin_temp"] = float(temp)

            if vitals_update and self.on_data_received:
                self.on_data_received(vitals_update)
        except Exception as e:
            print(f"[BLE] Error parsing GATT notification: {e}")

    async def connect_and_listen(self):
        """Scans for the wristband and maintains a connection to stream data."""
        self._running = True
        print(f"[BLE] Scanning for device with name '{self.target_name}'...")
        
        while self._running:
            try:
                device = await BleakScanner.find_device_by_filter(
                    lambda d, ad: d.name and self.target_name in d.name,
                    timeout=5.0
                )
                
                if not device:
                    print(f"[BLE] Target device '{self.target_name}' not found. Retrying in 10s...")
                    await asyncio.sleep(10.0)
                    continue

                print(f"[BLE] Found device: {device.name} [{device.address}]. Connecting...")
                
                async with BleakClient(device) as client:
                    self.client = client
                    self.is_connected = client.is_connected
                    print(f"[BLE] Connected to {device.name}!")
                    
                    # Subscribe to notifications
                    try:
                        await client.start_notify(HEART_RATE_MEASUREMENT_UUID, self._notification_handler)
                        # Optionally subscribe to custom fields if available on GATT server
                        await client.start_notify(SPO2_MEASUREMENT_UUID, self._notification_handler)
                        await client.start_notify(TEMP_MEASUREMENT_UUID, self._notification_handler)
                    except Exception as e:
                        print(f"[BLE] Warning, could not subscribe to all characteristics: {e}")
                        print(f"[BLE] Relying on Heart Rate characteristic notifications.")

                    # Keep connection alive
                    while client.is_connected and self._running:
                        await asyncio.sleep(1.0)
                        
                    print("[BLE] Connection lost.")
                    self.is_connected = False
                    
            except Exception as e:
                print(f"[BLE] Connection attempt encountered error: {e}")
                self.is_connected = False
                await asyncio.sleep(5.0)

    def stop(self):
        self._running = False
        self.is_connected = False
        print("[BLE] Bluetooth receiver stopped.")
