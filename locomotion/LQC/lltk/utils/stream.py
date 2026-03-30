# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import socket
import threading
import time

import numpy as np

from lltk.utils.sprint import sprint

__all__ = ['DataStream']


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.size > 100:
                return obj.flatten()[:100]
            if obj.dtype == np.float32:
                obj = obj.astype(float)
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class DataStream(object):
    def __init__(self, visualizer='PlotJuggler', dump=None):
        self._core = make_publisher(visualizer)
        self._count = 0
        self._dump = dump
        if self._dump is not None:
            if not self._dump.endswith('.json'):
                self._dump = f'{self._dump}.json'
            self._dump = open(self._dump, 'w')
        self._history_data = []

        self._offset = self._interval = None
        self._fixed_interval_mode = False

    def __del__(self):
        if self._dump is not None:
            sprint.bG(f"Dumping locomotion data to {self._dump.name} ...")
            json.dump(self._history_data, self._dump, cls=NumpyJsonEncoder)
            self._dump.close()

    def set_fixed_interval_mode(self, offset, interval):
        self._fixed_interval_mode = True
        self._offset = offset
        self._interval = interval
        self._count = 0

    def publish(self, data, timestamp=None):
        if timestamp is None:
            if self._fixed_interval_mode:
                timestamp = self._offset + self._count * self._interval
            else:
                timestamp = time.time()

        data = data | {'stamp': timestamp}
        self._core.send(data)
        if self._dump:
            self._history_data.append(data)
        self._count += 1


class Publisher:
    def __init__(self, *args, **kwargs):
        pass

    def send(self, data: dict):
        pass


def make_publisher(visualizer, *args, **kwargs) -> Publisher:
    visualizer = visualizer.lower()
    if visualizer == 'plotjuggler':
        return UdpPublisher(*args, **kwargs)
    if visualizer == 'foxglove':
        return FoxglovePublisher(*args, **kwargs)
    raise ValueError(f'Unknown visualizer `{visualizer}`')


class UdpPublisher(Publisher):
    """
    Send data stream of locomotion to outer tools such as PlotJuggler.
    """

    def __init__(self, port=9870):
        super().__init__()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.port = port

    def send(self, data: dict):
        msg = json.dumps(data, cls=NumpyJsonEncoder)
        ip_port = ('127.0.0.1', self.port)
        self.server.sendto(msg.encode('utf-8'), ip_port)


class SchemaGen:
    def __init__(self, obj=None):
        self.obj = obj

    def load(self, json_str):
        self.obj = json.loads(json_str)

    @staticmethod
    def default_obj():
        return {
            "type": "object",
            "properties": {},
        }

    def _skemate_list(self, obj):
        return {
            'type': 'array',
            'properties': [self._auto_skemate(value) for value in obj]
        }

    def _auto_skemate(self, obj):
        if isinstance(obj, dict):
            schema = self.default_obj()
            for key, value in obj.items():
                schema['properties'][key] = self._auto_skemate(value)
            return schema
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return {
                'type': 'array',
                'items': {'type': 'number'}
            }
        elif isinstance(obj, str):
            return {'type': 'string'}
        elif isinstance(obj, int):
            return {'type': 'integer'}
        elif isinstance(obj, float):
            return {'type': 'number'}
        return {'type': 'object'}

    def skemate(self):
        if isinstance(self.obj, dict):
            return self._auto_skemate(self.obj)
        schema = self.default_obj()
        schema['properties']['data'] = self._auto_skemate(self.obj)
        return schema


class FoxglovePublisher(Publisher):
    """
    Send data stream of locomotion to Foxglove.
    """

    def __init__(self, port=8765):
        super().__init__()
        self.server_lock = threading.Lock()
        self.server_thread = threading.Thread(
            target=lambda: asyncio.run(self.main(port)), daemon=True
        )
        self.server_lock.acquire()
        self.server_thread.start()
        while self.server_lock.locked():
            pass
        self.channels = {}

    def __del__(self):
        self.server.close()

    def send(self, data: dict):
        from foxglove_websocket.types import ChannelWithoutId

        self.server_lock.acquire()
        timestamp = time.time_ns()
        for k, v in data.items():
            if k not in self.channels:
                channel = ChannelWithoutId(
                    topic=k,
                    encoding='json',
                    schemaName=f'{k}MsgType',
                    schema=json.dumps(SchemaGen(v).skemate())
                )
                self.channels[k] = self._add_channel(channel)
            channel_id = self.channels[k]
            if not isinstance(v, dict):
                v = {'data': v}
            self._send_message(channel_id, timestamp, v)
        self.server_lock.release()

    def _add_channel(self, info):
        return asyncio.run(
            self.server.add_channel(info)
        )

    def _send_message(self, channel_id, timestamp, data):
        return asyncio.run(
            self.server.send_message(
                channel_id,
                timestamp,
                json.dumps(data, cls=NumpyJsonEncoder).encode("utf8"),
            )
        )

    async def main(self, port):
        from foxglove_websocket.server import FoxgloveServer
        self.server = FoxgloveServer("0.0.0.0", port, "data streamer")
        self.server.start()
        await self.server.wait_opened()
        self.server_lock.release()
        await self.server.wait_closed()
