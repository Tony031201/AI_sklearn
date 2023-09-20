/*
 * A4.c
 *
 * Created: 7/24/2023 2:29:37 PM
 * Author : weitingye
 */ 
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <avr/interrupt.h>
#include <avr/io.h>

#include "main.h"
#include "lcd_drv.h"

// These are included by the LCD driver code, so
// we don't need to include them here.
// #include <avr/io.h>
// #include <util/delay.h>

int hr=0;
int min=0;
int sec=0;
int subsec=0;
char*msg;
unsigned int high;	//used to store the button value
unsigned int low;	//used to store the button value
bool flag=false;	//used to check the button

//interrupt each 1/100 second, change the time 
ISR(TIMER3_COMPA_vect){
	subsec++;
	if(subsec>99){
		subsec=0;
		sec++;
	}
	
	if(sec>59)
	{
		sec = 0;
		min++;
	}

	if(min>59)
	{
		min = 0;
		hr++;
	}

	if(hr>24)

	{
		hr = 0;
	}

	sprintf(msg, "%02i:%02i:%02i:%02i", hr ,min, sec,subsec);	//update the time information

}

//check if the button is pressed or not
ISR(TIMER4_COMPA_vect){
	ADCSRA = ADCSRA | 0x40;

	while (ADCSRA & 0x40)
	{
		
	}

	low = ADCL;
	high = ADCH;

	low += (high << 8);
	
	//if the button is pressed, flag change
	if (low < 1000)
	{
		if (flag == false)
		{
			flag = true; 
		}

		else
		{
			flag = false;
		}
	}
}

int main( void )
{
	msg="00:00:00:00";
	//initialize the button
	ADCSRA = 0x87;
	ADMUX = 0x40;
	
	//initialize the timer4(ABOUT THE BUTTON)
	TCCR4B |=(1<<WGM42);					//set CTC mode
	TCCR4B |=(1<<CS41)|(1<<CS40);			
	OCR4A=2499;
	TIMSK4|=(1<<OCIE4A);
	
	//initialize the timer3(ABOUT THE CHANGING OF TIME)
	TCCR3B |=(1<<WGM32);					//set CTC mode
	TCCR3B |=(1<<CS31)|(1<<CS30);
	OCR3A=2499;
	TIMSK3|=(1<<OCIE3A);
	
	//GLOBAL INTERRUPT
	sei();
	
	//initialize the screen.
	lcd_init();
	
	//loop,is used to keep changing the screen.
	while(true){
		lcd_xy( 0, 0 );
		lcd_puts(msg);
		if(flag==false){
			lcd_xy(0,1);
			lcd_puts(msg);
		}
	}

	return 1;
}

