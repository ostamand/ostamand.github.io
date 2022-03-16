---
layout: post
title: "ESP8266 MicroPython Setup"
math: true
---

## Introduction

I used CircuitPython for my last project and I have to say I really enjoyed it. For my next one, I wanted to use some [Adafruit Feather HUZZAH with ESP8266](https://www.adafruit.com/product/2821) laying arround. Unfortunately, they are not compatible with CirtuitPython. Therefore, I decided to give a try at [MicroPython](http://www.micropython.org/).

This post is a concise summary of the development setup I am using on a mac.

Let's start.

## MicroPython Setup

Download MicroPython firmware ([link](http://micropython.org/download#esp8266)).

Create a virtual environment with esptool.

```
python3 -m venv venv
source venv/bin/activate
pip3 install esptool
```

Plug the board and get the usb port name needed to connect.

```
ls /dev/tty*
```

Erase the flash memory and deploy the new firmware:

```
esptool.py --port /dev/tty.SLAB_USBtoUART erase_flash
esptool.py --port /dev/tty.SLAB_USBtoUART --baud 460800 write_flash --flash_size=detect 0 bin/esp8266-20190125-v1.10.bin
```

Use `screen` as a terminal emulator to connect to the board now running MicroPython. You should see `>>>` indicating that you are now connected to the REPL.

```
screen /dev/tty.SLAB_USBtoUART 115200
```

Check if we can flash the LED attached to GPIO0.

```python
import machine
pin = machine.Pin(0, machine.Pin.OUT)
pin.on()
pin.off()
```

While still connected to the REPL, setup WebREPL.

```python
import webrepl_setup
```

Reboot the device and connect to the MicroPython-XXXXXX access point using the default password micropythoN.

Connect to the device using: http://micropython.org/webrepl/.

Use the "send a file" button to export files to the ESP8266. The `main.py` file will be automatically executed when the board loads up. For example, create a `main.py` file with the content below and export it to the board using WebREPL. After rebooting, a red led should again be flashing.

```python
import machine
import time

pin = machine.Pin(0, machine.Pin.OUT)

state = 0
while True:
    pin.value(state)
    state = not state
    time.sleep(1)
```

## Ampy Setup

Although using the WebREPL to transfer files does work it is somewhat cumbersome. I prefer to use ampy.

Install ampy.

```
pip install adafruit-ampy
```

Create a file named `test.py` with a print statement.

```python
print('Hello World!')
```

Run this file on the board using ampy. It will wait for all the code to be executed before displaying all the outputs to the console. To not wait before returning use the `--no-output` option (for example, useful in the case of a while loop). The running outputs can be accessed through a standard REPL screen.

```
ampy --port /dev/tty.SLAB_USBtoUART run test.py
```

To copy files on the board use the `put` command below. It can also transfer folders.

```
ampy --port /dev/tty.SLAB_USBtoUART put test.py
```

For more details on ampy refer to the Adafruit [documentation](https://learn.adafruit.com/micropython-basics-load-files-and-run-code/).
