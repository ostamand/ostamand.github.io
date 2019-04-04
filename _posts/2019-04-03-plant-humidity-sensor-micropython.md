---
layout: post
title:  "Plant Humidity Sensor with MicroPython & ESP8266"
categories: iot
math: true
---

# Introduction 

I always forget to water my plants and/or get confused when was the last time I did. No more. 

I will use a soil moisture sensor connected to [Adafruit IO](https://io.adafruit.com) to monitor the state of my plants in real-time. Additionnally, a one meter NeoPixel RGB strip will provide a visual feedback during the evening. That's when I do my watering and no need to waste power for leds in bright sunshine. 

For this project, I decided to use MicroPython running on a ESP8266. 

Below, I will summarize the main steps of the project as well as show the final result. It won't be a do it yourself step by step tutorial, more of a project journal where I summarize the development. I will include the full list of components, the board layout, the steps needed to setup MicroPython and all the code needed. 

# Components

List of all the components needed for this project.

- 1x [Adafruit Feather HUZZAH with ESP8266](https://www.adafruit.com/product/2821)
- 1x [NeoPixel RGB strip](https://www.adafruit.com/product/1460?length=1)
- 1x [Sparkfun Soil Moisture Sensor](https://www.sparkfun.com/products/13637)
    - VCC 3.3V-5.0V
- 1x 100k potentiomenter
- 1x 1k ohm resistor 
- 1x 10k ohm resistor 
- 1x 470 ohm resistor 
- 1x 1000 uF capacitor
- 2x SPDT slide switch
- 1x led  
- 1x diode
- 2x terminal block, 3 pins

# Details 

## Board

A layout is first done on a breadboard. 

A potentiometer is used to calibrate the moisture sensor signal. The ESP8266 ADC max input is 1.0V, therefore, depending on the type of soil and the desired mositure level, some tuning is required. 

One slide switch can be used to manually turn off the NeoPixel strip (in case I get tired of the NeoPixel lights and for, whatever reason, I don't want to water the plants).

A second slider switch is used to put the board in calibration mode (more on that later). Finally, a red led indicates when the power to the soil moisture is on. 

![layout_bb](/assets/original_layout_bb.png)

This breadboard layout is then put on a stripboard.

![layout_bb_stripboard](/assets/strip_board_layout_bb.png)

The final board, after soldering and testing is shown below. 



