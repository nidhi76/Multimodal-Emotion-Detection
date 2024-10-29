from ast import Lambda
from tkinter import Button, Canvas, PhotoImage, Tk, TkVersion
import tkinter
from tkinter.ttk import Label
from turtle import bgcolor, color, onclick
from PIL import Image, ImageTk
import cv2
from numpy import size
from sklearn import ensemble
import ensemble
from threading import*


def Threading():
    t1 = Thread(target=Webcamera)
    t2 = Thread(target=call_to_text)
    t1.start()
    t2.start()



def Webcamera():
    global A
    A = ensemble.image_analysis()
    
def call_to_text():

    def printInput():
        global B
        text = inputtxt.get()
        #print(type(text))
        print(text)
        B = ensemble.text_analysis(text)
  
    frame = Tk()
    frame.title("Text window")
    w=500
    h=150
    x=820
    y=400
    #frame.geometry("300x150")
    frame.geometry('%dx%d+%d+%d'%(w,h,x,y))
    frame.config(bg='violet')

    printButton = Button(frame,
                        text = "Add", 
                        command= printInput)
    
    printButton.pack()
        
    inputtxt = tkinter.Entry(frame, width= 70,font = ('calibre',10,'normal'))
    inputtxt.pack()

    frame.after(25000,lambda:frame.destroy())

    frame.mainloop()


def startmusic():
     ensemble.ensemble(A,B)



    


root = Tk()

#bg = PhotoImage(file = "music_wallpaper3.png")	
bg =Image.open("music_wallpaper3.png")	
resized=bg.resize((1900,1200),Image.ANTIALIAS)
bg2 = ImageTk.PhotoImage(resized)
root.title("Music Player App")

txt = tkinter.StringVar()

root.geometry("1500x1200")

root.config(bg='gray')
canvas = Canvas(root, width = 1500,
                 height = 1200)

canvas.pack(fill = "both", expand = True)
canvas.create_image( 0, 0, image = bg2, 
                     anchor = "nw")
  

label = Label(canvas,text="Music Player",font=("Arial", 40),background='violet').pack(pady=20)

#Btn1 = Button(text="Take Image",background="blue",font=70,command=Webcamera)
#Btn1.place(x=120,y=150)

#Btn2 = Button(text="Add Text",background="blue",font=70,command=call_to_text)
#Btn2.place(x=750,y=150)

Btn3 = Button(text="Start Music",background="blue",font=80,command=startmusic)
Btn3.place(x=1200,y=250)

button= Button(root, text= "Capture Emotion",background="blue",font=80,command= Threading)
button.place(x=300,y=250)




	
root.mainloop()
