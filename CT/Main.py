from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from Classification_Tree import predict_by_clt_model
from SVR.SVR_model import predect_SVR
from ANN.ANN import predict_ANN


def centring_the_window(width, height, cur_root):
    screen_width = cur_root.winfo_screenwidth()
    screen_height = cur_root.winfo_screenheight()

    x_root = (screen_width - width) // 2
    y_root = (screen_height - height) // 2

    cur_root.geometry(f'{width}x{height}+{x_root}+{y_root}')


def main_start():
    main_root = Tk()

    def open_clt_page():
        main_root.destroy()
        clt_and_nn_start('Classification Tree')

    def open_nn_page():
        main_root.destroy()
        clt_and_nn_start('Neural Network')

    def open_svm_page():
        main_root.destroy()
        svm_start()

    # ------------ Setting some properties of the root ------------ #
    main_root.title('AML Project')
    main_root.resizable(False, False)
    main_root.iconbitmap('images/icon.ico')

    # ------------ Centering the window ------------ #
    centring_the_window(430, 450, main_root)

    # ------------ Placing The logo ------------ #
    logo = PhotoImage(file='images/main_logo.png').subsample(3, 5)
    logo_label = Label(image=logo)
    logo_label.pack()

    # ------------ Placing The buttons ------------ #
    buttons_frame = Frame(main_root)
    buttons_frame.pack(pady=25)

    button_style = {
        'fg': 'white smoke',
        'bg': 'DarkOrange1',
        'font': ('Comic Sans MS', 12, 'bold'),
        'relief': 'groove',
        'borderwidth': 2
    }
    btn_svm = Button(buttons_frame, text='Support Vector Machine', width=20, **button_style, command=open_svm_page)
    btn_clt = Button(buttons_frame, text='Classification Tree', width=20, **button_style, command=open_clt_page)
    btn_nn = Button(buttons_frame, text='Neural Network', width=20, **button_style, command=open_nn_page)

    btn_svm.pack(pady=10)
    btn_clt.pack(pady=10)
    btn_nn.pack(pady=10)

    # ------------ Displaying the window ------------ #
    main_root.mainloop()


def clt_and_nn_start(model):
    clt_root = Tk()

    def back_to_home():
        clt_root.destroy()
        main_start()

    def predict_result():
        # Updating the message with every button click #
        for widget in result_frame.winfo_children():
            widget.destroy()

        # Checking for empty fields #
        info = {'Education': ed_cb.get(), 'JoiningYear': jy_en.get(), 'City': ct_cb.get(), 'PaymentTier': pt_cb.get(),
                'Age': age_en.get(), 'Gender': gn_cb.get(), 'EverBenched': eb_cb.get(),
                'ExperienceInCurrentDomain': exp_en.get()}
        for val in info.values():
            if len(val) == 0:
                messagebox.showinfo('Error', 'You should fill all the fields.')
                return

        # Checking for invalid inputs #
        try:
            info['JoiningYear'] = int(info['JoiningYear'])
            info['Age'] = int(info['Age'])
            info['ExperienceInCurrentDomain'] = int(info['ExperienceInCurrentDomain'])
            info['PaymentTier'] = int(info['PaymentTier'])
        except:
            messagebox.showinfo('Error', 'Joining year, Age and Experience must be of integer type.')
            return

        err_msg = ''
        err_hap = False
        if info['JoiningYear'] < 1974 or info['JoiningYear'] > 2024:
            err_msg += 'Joining year must be between 1974 and 2024.\n'
            err_hap = True
        if info['Age'] < 20 or info['Age'] > 70:
            err_msg += 'Age must be between 20 and 70.\n'
            err_hap = True
        if err_hap:
            messagebox.showinfo('Error', err_msg)
            return

        # Placing the predicted result #
        if model == 'Classification Tree':
            rs = predict_by_clt_model(info)
        else:
            rs = predict_ANN(info)

        msg = 'The employee should ' + ('stay.' if rs == 0 else 'leave.')
        rs_str = Message(result_frame, width=500, text=msg,
                         font=('Comic Sans MS', 13), bg=frames_color)
        rs_str.pack()

    # ------------ Setting some properties of the root ------------ #
    page_title = 'Classification Tree' if model == 'Classification Tree' else 'Neural Network'
    clt_root.title(page_title)
    clt_root.resizable(False, False)
    clt_root.iconbitmap('images/icon.ico')
    frames_color = 'ghost white'
    clt_root.config(bg=frames_color)

    # ------------ Centering the window ------------ #
    centring_the_window(860, 500, clt_root)

    # ------------ Placing The logo ------------ #
    img_path = 'images/clt_logo.png' if model == 'Classification Tree' else 'images/nn_logo.png'
    logo = PhotoImage(file=img_path).subsample(2, 2)
    logo_label = Label(image=logo)
    logo_label.pack(side='right')

    # ------------ Placing the top frame ------------ #
    top_frame = Frame(clt_root, height=30, bg=frames_color)
    top_frame.pack(fill=BOTH, side='top')

    back_icon = PhotoImage(file='images/back_icon_2.png').subsample(3)
    back_btn = Button(top_frame, image=back_icon, relief='flat', bg=frames_color, command=back_to_home)
    back_btn.grid(row=0, column=0)

    emp_icon = PhotoImage(file='images/emp_icon.png').subsample(3)
    emp_lbl = Label(top_frame, image=emp_icon, bg=frames_color)
    emp_lbl.grid(row=0, column=1, padx=99)

    # ------------ Placing The inputs ------------ #
    inputs_frame = LabelFrame(clt_root, text='Info', width=400, bg=frames_color)
    # inputs_frame.place(x=18, y=80)
    inputs_frame.pack(fill=BOTH, padx=10, pady=10)

    label_style = {
        # "fg": 'DarkOrange2',
        'font': ('Comic Sans MS', 14),
        'bg': frames_color
    }

    en_pad_x_val = 5
    ed_lb = Label(inputs_frame, text='Education', **label_style)
    ed_lb.grid(row=0, column=0)
    ed_cb = ttk.Combobox(inputs_frame, width=27, state='readonly', values=['Bachelors', 'Masters', 'PHD'])
    ed_cb.grid(row=0, column=1, padx=en_pad_x_val)

    jy_lb = Label(inputs_frame, text='JoiningYear', **label_style)
    jy_lb.grid(row=1, column=0)
    jy_en = Entry(inputs_frame, width=30)
    jy_en.grid(row=1, column=1, padx=en_pad_x_val)

    ct_lb = Label(inputs_frame, text='City', **label_style)
    ct_lb.grid(row=2, column=0)
    ct_cb = ttk.Combobox(inputs_frame, width=27, state='readonly', values=['Bangalore', 'Pune', 'New Delhi'])
    ct_cb.grid(row=2, column=1, padx=en_pad_x_val)

    pt_lb = Label(inputs_frame, text='PaymentTier', **label_style)
    pt_lb.grid(row=3, column=0)
    pt_cb = ttk.Combobox(inputs_frame, width=27, state='readonly', values=['1', '2', '3'])
    pt_cb.grid(row=3, column=1, padx=en_pad_x_val)

    age_lb = Label(inputs_frame, text='Age', **label_style)
    age_lb.grid(row=4, column=0)
    age_en = Entry(inputs_frame, width=30)
    age_en.grid(row=4, column=1, padx=en_pad_x_val)

    gn_lb = Label(inputs_frame, text='Gender', **label_style)
    gn_lb.grid(row=5, column=0)
    gn_cb = ttk.Combobox(inputs_frame, width=27, state='readonly', values=['Male', 'Female'])
    gn_cb.grid(row=5, column=1, padx=en_pad_x_val)

    eb_lb = Label(inputs_frame, text='EverBenched', **label_style)
    eb_lb.grid(row=6, column=0)
    eb_cb = ttk.Combobox(inputs_frame, width=27, state='readonly', values=['No', 'Yes'])
    eb_cb.grid(row=6, column=1, padx=en_pad_x_val)

    exp_lb = Label(inputs_frame, text='Experience', **label_style)
    exp_lb.grid(row=7, column=0)
    exp_en = Entry(inputs_frame, width=30)
    exp_en.grid(row=7, column=1, padx=en_pad_x_val)

    # ------------ Placing the predict button ------------ #
    prd_btn = Button(text='Predict', fg='white smoke', bg='DarkOrange2', width=10,
                     font=('Comic Sans MS', 12, 'bold'), relief='groove', command=predict_result)
    prd_btn.place(x=120, y=370)

    result_frame = LabelFrame(clt_root, text='Result', bg=frames_color)
    result_frame.pack(fill=BOTH, padx=10, pady=46)

    # ------------ Displaying the window ------------ #
    clt_root.mainloop()


def svm_start():
    svm_root = Tk()

    def back_to_home():
        svm_root.destroy()
        main_start()

    def predict():
        # Updating the message with every button click #
        for widget in result_frame.winfo_children():
            widget.destroy()

        info = {'contest1': contest_1_en.get(), 'contest2': contest_2_en.get(),
                'contest3': contest_3_en.get(), 'contest4': contest_4_en.get(), 'contest5': contest_5_en.get(),
                'contest6': contest_6_en.get(), 'contest7': contest_7_en.get(), 'contest8': contest_8_en.get(),
                'contest9': contest_9_en.get(), 'contest10': contest_10_en.get()}
        for key in info.keys():
            # Checking for empty fields #
            if len(info[key]) == 0:
                messagebox.showinfo('Error', 'You should fill all the fields.')
                return

            # Checking for invalid inputs #
            try:
                info[key] = int(info[key])
            except:
                messagebox.showinfo('Error', 'All the fields must be of integer type.')
                return

            if info[key] < 0:
                messagebox.showinfo('Error', 'All the fields must be positive.')
                return

        # Predicting the result #
        rs = predect_SVR(info)
        Label(result_frame, text=f'Rank predicted: {rs}', **label_style).pack()

    # ------------ Setting some properties of the root ------------ #
    svm_root.title('Support Vector Machine')
    svm_root.resizable(False, False)
    svm_root.iconbitmap('images/icon.ico')
    frames_color = 'ghost white'
    svm_root.config(bg=frames_color)

    # ------------ Centering the window ------------ #
    centring_the_window(890, 570, svm_root)

    # ------------ Placing The logo ------------ #
    logo = PhotoImage(file='images/svm_logo_3.png')
    logo_label = Label(image=logo, bg=frames_color)
    logo_label.pack(side='right')

    # ------------ Placing the top frame ------------ #
    top_frame = Frame(svm_root, height=30, bg=frames_color)
    top_frame.pack(fill=BOTH, side='top')

    back_icon = PhotoImage(file='images/back_icon_2.png').subsample(3)
    back_btn = Button(top_frame, image=back_icon, relief='flat', bg=frames_color, command=back_to_home)
    back_btn.grid(row=0, column=0)

    cf_icon = PhotoImage(file='images/cf_icon_3.png')
    cf_lbl = Label(top_frame, image=cf_icon, bg=frames_color)
    cf_lbl.grid(row=0, column=1, padx=25)

    # ------------ Placing The inputs ------------ #
    inputs_frame = LabelFrame(svm_root, text='Info', bg=frames_color)
    inputs_frame.pack(padx=30, pady=10, ipadx=20)

    label_style = {
        # "fg": 'DarkOrange2',
        'font': ('Comic Sans MS', 14),
        'bg': frames_color
    }

    lb_pad_x_val = 10

    contest_1_lb = Label(inputs_frame, text='Contest 1', **label_style)
    contest_1_lb.grid(row=1, column=0, padx=lb_pad_x_val)
    contest_1_en = Entry(inputs_frame)
    contest_1_en.grid(row=1, column=1)

    contest_2_lb = Label(inputs_frame, text='Contest 2', **label_style)
    contest_2_lb.grid(row=2, column=0, padx=lb_pad_x_val)
    contest_2_en = Entry(inputs_frame)
    contest_2_en.grid(row=2, column=1)

    contest_3_lb = Label(inputs_frame, text='Contest 3', **label_style)
    contest_3_lb.grid(row=3, column=0, padx=lb_pad_x_val)
    contest_3_en = Entry(inputs_frame)
    contest_3_en.grid(row=3, column=1)

    contest_4_lb = Label(inputs_frame, text='Contest 4', **label_style)
    contest_4_lb.grid(row=4, column=0, padx=lb_pad_x_val)
    contest_4_en = Entry(inputs_frame)
    contest_4_en.grid(row=4, column=1)

    contest_5_lb = Label(inputs_frame, text='Contest 5', **label_style)
    contest_5_lb.grid(row=5, column=0, padx=lb_pad_x_val)
    contest_5_en = Entry(inputs_frame)
    contest_5_en.grid(row=5, column=1)

    contest_6_lb = Label(inputs_frame, text='Contest 6', **label_style)
    contest_6_lb.grid(row=6, column=0, padx=lb_pad_x_val)
    contest_6_en = Entry(inputs_frame)
    contest_6_en.grid(row=6, column=1)

    contest_7_lb = Label(inputs_frame, text='Contest 7', **label_style)
    contest_7_lb.grid(row=7, column=0, padx=lb_pad_x_val)
    contest_7_en = Entry(inputs_frame)
    contest_7_en.grid(row=7, column=1)

    contest_8_lb = Label(inputs_frame, text='Contest 8', **label_style)
    contest_8_lb.grid(row=8, column=0, padx=lb_pad_x_val)
    contest_8_en = Entry(inputs_frame)
    contest_8_en.grid(row=8, column=1)

    contest_9_lb = Label(inputs_frame, text='Contest 9', **label_style)
    contest_9_lb.grid(row=9, column=0, padx=lb_pad_x_val)
    contest_9_en = Entry(inputs_frame)
    contest_9_en.grid(row=9, column=1)

    contest_10_lb = Label(inputs_frame, text='Contest 10', **label_style)
    contest_10_lb.grid(row=10, column=0, padx=lb_pad_x_val)
    contest_10_en = Entry(inputs_frame)
    contest_10_en.grid(row=10, column=1)

    # ------------ Placing the predict button ------------ #
    prd_btn = Button(text='Predict', fg='white smoke', bg='DarkOrange2', width=10,
                     font=('Comic Sans MS', 12, 'bold'), relief='groove', command=predict)
    prd_btn.place(x=100, y=440)

    result_frame = LabelFrame(svm_root, text='Result', bg=frames_color)
    result_frame.place(x=38, y=490)

    # ------------ Displaying the window ------------ #
    svm_root.mainloop()


main_start()
# clt_and_nn_start()
# svm_start()
