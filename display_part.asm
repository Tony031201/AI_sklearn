#define LCD_LIBONLY

.include "lcd.asm"

.cseg

#define DELAY 1.2
#define DELAY2 1


#define CLOCK 16.0e6
.equ PRESCALE_DIV=1024  ; implies CS[2:0] is 0b101

;Write timer below:
.equ dly=int(0.5+(CLOCK/PRESCALE_DIV*DELAY))
.if dly>65535
.error "dly is out of range"
.endif

.equ dly2=int(0.5+(CLOCK/PRESCALE_DIV*DELAY2))
.if dly2>65535
.error "dly2 is out of range"
.endif

;First Initilize built-in Analog to Digital Converter
; initialize the Analog to Digital converter
	ldi r16, 0x87
	sts ADCSRA, r16
	ldi r16, 0x40
	sts ADMUX, r16

	; set Timer 3
	;
	ldi r17, high(dly)
	ldi r16, low(dly)
	sts OCR3AH, r17
	sts OCR3AL, r16

	ldi r19, 0
	sts TCCR3A, r19

	ldi r19, (1 << WGM32) | (1 << CS32) | (1 << CS30)
	sts TCCR3B, r19

	;set Timer 4
	;
	ldi r17, high(dly2)
	ldi r16, low(dly2)
	sts OCR4AH, r17
	sts OCR4AL, r16

	ldi r19, 0
	sts TCCR4A, r19

	ldi r19, (1 << WGM42) | (1 << CS42) | (1 << CS40)
	sts TCCR4B, r19


	
str_length:			;str length: caculate the string length to know how long should I scroll
	;push the value to store
	push r16
	push r18
	push r30
	push r31

	clr r17			;I use r17 to store the length of msg1
	ldi r16, high(msg1_p<<1)
	mov r31, r16
	ldi r16, low(msg1_p<<1)
	mov r30, r16
	;start to record the length
	start:
	lpm r18,Z+
	cpi r18,0x00
	brne plus
	rjmp finish		 ;once finish to record, jump to the next string
	plus:			 ;record the length by increase r17
	inc r17
	rjmp start		;loop
	;the end of the loop
	finish: 
	

	clr r23			;use r23 to store the length of msg2
	ldi r16, high(msg2_p<<1)
	mov r31, r16
	ldi r16, low(msg2_p<<1)
	mov r30, r16
	start2:
	lpm r18,Z+
	cpi r18,0x00
	brne plus2
	rjmp finish2
	plus2:
	inc r23
	rjmp start2
	finish2:
	;store the original value by pop the stack
	pop r31
	pop r30
	pop r18
	pop r16

	;r18 and r22 I will use to be the index of string1 and 2
	ldi r18,0
	ldi r22,0
	;use r24 to record the display status
	ldi r24,0
	;r26 I will use to determinate the diretion of scrolling. when it equal to 0,scroll right. if not, scroll left
	ldi r26,0

	call lcd_init			; call lcd_init to Initialize the LCD
	call init_strings		;call init_strings to store the strings into data memory

begin:						;main loop
	call check_button		;polling: each loop check if the button is pressed or not
	in r19,TIFR3			;check the timer3 which controll the scroll speed
	sbrc r19,OCF3A			
	call check_dir			;check the direction and scroll if the timer reach the TOP  value

	in r19,TIFR4
	sbrc r19,OCF4A
	call blink				;blink if the timer4 reach the TOP  value

	rjmp begin




lp:	jmp lp

;use for change the direction of scrolling
check_dir:
		cpi r26,0			;r26=0 if the left button is pressed
		breq leftscroll		;then scroll left
		rjmp rightscroll	;r26!=0 if the right button is pressed
		rightscroll:		
		call rescroll
		rjmp got			;got the direction
		leftscroll:
		call scroll
		got:				;the end of check_dir, it means the direction has been got.
		ret

;check which button is pressed
check_button:
		push r16
		push r17
		push r19
		; start a2d conversion
		lds	r16, ADCSRA	  ; get the current value of SDRA
		ori r16, 0x40     ; set the ADSC bit to 1 to initiate conversion
		sts	ADCSRA, r16

		; wait for A2D conversion to complete
	wait:	lds r16, ADCSRA
		andi r16, 0x40     ; see if conversion is over by checking ADSC bit
		brne wait          ; ADSC will be reset to 0 is finished

		; read the value avialble as 10 bits in ADCH:ADCL
		lds r16, ADCL
		lds r17, ADCH

	check_RIGHT:			;check the right button is pressed. if yes, set r26 to 0
		clr r19
		;start to check the value from button we get
		cpi r16, 0x32
		breq high1
		rjmp finish1
		high1:
		cpi r17,0
		breq true1
		rjmp finish1
		true1:
		ldi r26,1			;set the r26 to 0 if the right button be pressed
		ldi r19,1			;if we are surly know the right button is pressed, we set the r19 to 1
		finish1:
		nop
	
	check_LEFT:				;check the left button is pressed. if yes, set r26 to 1
		cpi r19,1			;if r19=1, it means the right button is pressed. WE DON'T NEED to check the left button.
		breq finishnew
		;start to check the value from button we get
		high2:
		cpi r17,0x02
		breq true2
		rjmp finish2new
		true2:
		ldi r26,0			;;set the r26 to 1 if the left button be pressed
		finish2new:
		ldi r19,0			;clear r19
		nop

	finishnew:
		;end of check_button
		pop r19
		pop r17
		pop r16
		ret


blink:

	; This subroutine controll the blink of the lcd

	ldi r19, 1<<OCF4A		;clear bit 1 in TIFR4 by writing logical one to its bit position, P163 of the Datasheet
	out TIFR4, r19

	cpi r24,0				;if r24=0,it means now is the beginning, and jump to case1
	breq case1
	cpi r24,1				;if r24=1,it means now is the 2nd second, and jump to case2
	breq case2
	cpi r24,2				;if r24=2,it means now is the 3rd second, and jump to case3
	breq case3				
	cpi r24,3				;if r24=3,it means we finish one iteration, reset to the begining
	breq case4
	case1:					;the beginning of lcd blink,each line should be display
	call lcd_clr 
	call display_strings1
	call display_strings2
	rjmp end3
	case2:					;the 2nd second,display the first line AND turn off the second line
	call lcd_clr
	call display_strings1
	rjmp end3
	case3:					;the 3rd second,display the second line AND turn off the first line
	call lcd_clr
	call display_strings2
	rjmp end3
	case4:
	clr r24
	rjmp case1
	end3:
	inc r24					;after one blink, r24 increase which means go into the next step
	ret


init_strings:
	push r16
	; copy strings from program memory to data memory
	ldi r16, high(msg1)		; this the destination
	push r16
	ldi r16, low(msg1)
	push r16
	ldi r16, high(msg1_p << 1) ; this is the source
	push r16
	ldi r16, low(msg1_p << 1)
	push r16
	call str_init			; copy from program to data
	pop r16					; remove the parameters from the stack
	pop r16
	pop r16
	pop r16

	ldi r16, high(msg2)
	push r16
	ldi r16, low(msg2)
	push r16
	ldi r16, high(msg2_p << 1)
	push r16
	ldi r16, low(msg2_p << 1)
	push r16
	call str_init
	pop r16
	pop r16
	pop r16
	pop r16

	pop r16
	ret

display_strings1:

	; This subroutine display the string1

	push r16

	; Now move the cursor to the first line (ie. 0,0)
	ldi r16, 0x00
	push r16
	ldi r16, 0x00
	push r16
	call lcd_gotoxy
	pop r16
	pop r16

	; Now display msg1 on the first line
	ldi r16, high(msg1)
	push r16
	ldi r16, low(msg1)
	add r16,r18	
	push r16
	call lcd_puts
	pop r16
	pop r16

	pop r16 
	ret


display_strings2:

	; This subroutine display the string2

	push r16

	; Now move the cursor to the second line (ie. 0,1)
	ldi r16, 0x01
	push r16
	ldi r16, 0x00
	push r16
	call lcd_gotoxy
	pop r16
	pop r16

	; Now display msg2 on the second line
	ldi r16, high(msg2)
	push r16
	ldi r16, low(msg2)
	add r16,r22
	push r16
	call lcd_puts
	pop r16
	pop r16

	pop r16
	ret

	
scroll:

	; This subroutine use for scrolling right 

	ldi r19, 1<<OCF3A ;clear bit 1 in TIFR3 by writing logical one to its bit position, P163 of the Datasheet
	out TIFR3, r19

	;increase the index of string, it means changed the position I need to display
	;i.e. before that, the lcd display 'ABCD' , but after that it gonna display 'BCD'
	inc r18
	inc r22

	;check if the index reach the maximum of the length, if yes then clear to 0(restart)
	compare18:
	cp r18,r17
	breq clear18
	end18:
	nop
	compare22:
	cp r22,r23
	breq clear22
	end22:

	rjmp end2

	clear18:
	clr r18
	rjmp end18

	clear22:
	clr r22
	rjmp end22
	end2:					;end of scroll
	nop
	ret

rescroll:

	; This subroutine use for scrolling left 
	;similiar to scroll, but the index will be decrease

	ldi r19, 1<<OCF3A ;clear bit 1 in TIFR3 by writing logical one to its bit position, P163 of the Datasheet
	out TIFR3, r19

	dec r18
	dec r22

	Compare1:
	cpi r18,0
	breq set1
	end1new:
	nop
	Compare2:
	cpi r22,0
	breq set2
	end2new:

	rjmp end289

	set1:
	mov r18,r17
	rjmp end1new

	set2:
	mov r22,r23
	rjmp end2new

	end289:			;end of rescroll
	nop
	ret

	

	


msg1_p:	.db " Weiting Ye", 0	,0
msg2_p: .db " CSC 230: Summer 2023", 0 

.dseg
;
; The program copies the strings from program memory
; into data memory.  These are the strings
; that are actually displayed on the lcd
;
msg1:	.byte 200
msg2:	.byte 200

