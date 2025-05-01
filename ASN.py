# coret2_asn_gui.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import plotly.graph_objs as go
import math

# Fungsi utama
def main():
    st.title("ANALISIS SINYAL ECG DAN PERNAPASAN")
    st.sidebar.header("Pengaturan Parameter")
    
    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload file data", type=["txt"])
    
    if uploaded_file is not None:
        # Baca data
        data = pd.read_csv(uploaded_file, delimiter=r'\s+', header=None)
        
        # Proses waktu
        def time(waktu_str):
            m, s = waktu_str.split(':')
            return int(m) * 60 + float(s)
        
        waktu = data[0]
        x = data[5]
        resp = data[1]
        waktu = [time(waktu_str) for waktu_str in waktu]

        # Tab untuk berbagai visualisasi
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Sinyal Raw", 
            "DWT Levels", 
            "Processing Steps",
            "BPM Results",
            "Respiratory Comparison",
            "Time Domain Analysis",
            "Freq. Domain Analysis",
            "Linear Analysis"
        ])

        with tab1:
            st.header("Sinyal Raw")
            
            # Membuat grafik interaktif untuk ECG
            fig_ecg = go.Figure()
            fig_ecg.add_trace(go.Scatter(x=waktu, y=x, mode='lines', name='ECG Signal', line=dict(color='blue')))
            fig_ecg.update_layout(title='ECG Signal', xaxis_title='Waktu', yaxis_title='Amplitudo', dragmode='zoom')

            # Membuat grafik interaktif untuk Respiratory
            fig_resp = go.Figure()
            fig_resp.add_trace(go.Scatter(x=waktu, y=resp, mode='lines', name='Respiratory Signal', line=dict(color='magenta')))
            fig_resp.update_layout(title='Respiratory Signal', xaxis_title='Waktu', yaxis_title='Amplitudo', dragmode='zoom')

            # Menampilkan grafik di Streamlit
            st.plotly_chart(fig_ecg, use_container_width=True)
            st.plotly_chart(fig_resp, use_container_width=True)

            
            # Fungsi untuk meniru fungsi Dirac delta
            def dirac(x):
                if x == 0:
                    return 1
                else:
                    return 0

            # Hitung h(n) dan g(n)
            h = []
            g = []
            n_list = []
            for n in range(-2, 2):
                n_list.append(n)
                temp_h = (1/8) * (dirac(n-1) + 3 * dirac(n) + 3 * dirac(n+1) + dirac(n+2))
                h.append(temp_h)
                temp_g = -2 * (dirac(n) - dirac(n+1))
                g.append(temp_g)

            st.subheader("Impulse Responses h(n) dan g(n)")
            # Plot h(n)
            fig_h, ax_h = plt.subplots(figsize=(8, 4))
            ax_h.bar(n_list, h, width=0.1, color='green')
            ax_h.set_title("h(n)")
            st.pyplot(fig_h)
            
            # Plot g(n)
            fig_g, ax_g = plt.subplots(figsize=(8, 4))
            ax_g.bar(n_list, g, width=0.1, color='red')
            ax_g.set_title("g(n)")
            st.pyplot(fig_g)
            
            # Compute Hw dan Gw
            # Karena kita hanya memerlukan data hingga fs, kita buat array dengan panjang fs+1
            fs=125
            Hw = np.zeros(fs + 1)
            Gw = np.zeros(fs + 1)
            i_list = []
            for i in range(0, fs + 1):
                i_list.append(i)
                reG, imG = 0, 0
                reH, imH = 0, 0

                # Akumulasi nilai untuk reG, imG, reH, imH
                for k in range(-2, 2):
                    # Gunakan index offset: k=-2 => index 0, k=-1 => index 1, k=0 => index 2, k=1 => index 3
                    index = k + 2
                    reG += g[index] * np.cos(k * 2 * np.pi * i / fs)
                    imG -= g[index] * np.sin(k * 2 * np.pi * i / fs)
                    reH += h[index] * np.cos(k * 2 * np.pi * i / fs)
                    imH -= h[index] * np.sin(k * 2 * np.pi * i / fs)

                # Hitung magnitudo
                Hw[i] = np.sqrt(reH**2 + imH**2)
                Gw[i] = np.sqrt(reG**2 + imG**2)

            half_length = round(fs / 2) + 1
            i_list_half = i_list[0:half_length]

            st.subheader("Magnitude Hw dan Gw")
            # Plot Hw
            fig_Hw, ax_Hw = plt.subplots(figsize=(8, 4))
            ax_Hw.plot(i_list_half, Hw[0:half_length], label="Hw")
            ax_Hw.set_title("Hw")
            st.pyplot(fig_Hw)
            
            # Plot Gw
            fig_Gw, ax_Gw = plt.subplots(figsize=(8, 4))
            ax_Gw.plot(i_list_half, Gw[0:half_length], color='orange', label="Gw")
            ax_Gw.set_title("Gw")
            st.pyplot(fig_Gw)
            
            # Buat array 2D Q
            Q = np.zeros((9, half_length))
            for i in range(half_length):
                Q[0, i] = Gw[i]  # Q1
                if i > 0:
                    if 2 * i < len(Gw):
                        Q[1, i] = Gw[2 * i] * Hw[i]
                    if 4 * i < len(Gw):
                        Q[2, i] = Gw[4 * i] * Hw[2 * i] * Hw[i]
                    if 8 * i < len(Gw):
                        Q[3, i] = Gw[8 * i] * Hw[4 * i] * Hw[2 * i] * Hw[i]
                    if 16 * i < len(Gw):
                        Q[4, i] = Gw[16 * i] * Hw[8 * i] * Hw[4 * i] * Hw[2 * i] * Hw[i]
                    if 32 * i < len(Gw):
                        Q[5, i] = Gw[32 * i] * Hw[16 * i] * Hw[8 * i] * Hw[4 * i] * Hw[2 * i] * Hw[i]
                    if 64 * i < len(Gw):
                        Q[6, i] = Gw[64 * i] * Hw[32 * i] * Hw[16 * i] * Hw[8 * i] * Hw[4 * i] * Hw[2 * i] * Hw[i]
                    if 128 * i < len(Gw):
                        Q[7, i] = Gw[128 * i] * Hw[64 * i] * Hw[32 * i] * Hw[16 * i] * Hw[8 * i] * Hw[4 * i] * Hw[2 * i] * Hw[i]

            st.subheader("Magnitude")
            fig_Q, ax_Q = plt.subplots(figsize=(10, 6))
            ax_Q.plot(range(half_length), Q[0], label="Q1")
            for idx in range(1, 8):
                ax_Q.plot(range(half_length), Q[idx], label=f"Q{idx + 1}")
            ax_Q.legend()
            ax_Q.set_xlabel("fs/2")
            ax_Q.set_ylabel("Magnitude")
            ax_Q.set_title("Magnitude")
            ax_Q.grid()
            st.pyplot(fig_Q)

        with tab2:
            st.header("Analisis DWT")
            
            # Implementasi DWT Level 1-3 (disimplifikasi)
            def dirac(x):
                if x == 0:
                    return 1
                else:
                    return 0
            qj = np.zeros((9, len(x)))

            # ================================
            # DWT Level-1 (manual filter bank)
            # ================================
            k_list1 = []
            j = 1
            a = -(round(2**j) + round(2**(j-1)) - 2)
            print('Level-1 a =', a)
            b = -(1 - round(2**(j-1))) + 1
            print('Level-1 b =', b)

            for k in range(a, b):
                k_list1.append(k)
                qj[1][k + abs(a)] = -2 * (dirac(k) - dirac(k + 1))

            impulse_response_1 = qj[1][:len(k_list1)]
            detail_coeffs_1 = np.convolve(x, impulse_response_1, mode='same')

            # ================================
            # DWT Level-2 (manual filter bank)
            # ================================
            k_list2 = []
            j = 2
            a = -(round(2**j) + round(2**(j-1)) - 2)
            print('Level-2 a =', a)
            b = -(1 - round(2**(j-1))) + 1
            print('Level-2 b =', b)

            for k in range(a, b):
                k_list2.append(k)
                qj[2][k + abs(a)] = -1/4 * (
                    dirac(k - 1) + 3*dirac(k) + 2*dirac(k + 1)
                    - 2*dirac(k + 2) - 3*dirac(k + 3) - dirac(k + 4)
                )

            impulse_response_2 = qj[2][:len(k_list2)]
            detail_coeffs_2 = np.convolve(x, impulse_response_2, mode='same')

            # ================================
            # DWT Level-3 (manual filter bank)
            # ================================
            k_list3 = []
            j = 3
            a = -(round(2**j) + round(2**(j-1)) - 2)
            print('Level-3 a =', a)
            b = -(1 - round(2**(j-1))) + 1
            print('Level-3 b =', b)

            for k in range(a, b):
                k_list3.append(k)
                qj[3][k + abs(a)] = -1/32*(dirac(k-3) + 3*dirac(k-2) + 6*dirac(k-1) + 10*dirac(k) + 11*dirac(k+1)
                            + 9*dirac(k+2) + 4*dirac(k+3) - 4*dirac(k+4) - 9*dirac(k+5) - 11*dirac(k+6)
                            - 10*dirac(k+7) - 6*dirac(k+8) - 3*dirac(k+9) - dirac(k+10))

            impulse_response_3 = qj[3][:len(k_list3)]
            detail_coeffs_3 = np.convolve(x, impulse_response_3, mode='same')

            # ================================
            # DWT Level-4 (manual filter bank)
            # ================================
            k_list4 = []
            j = 4
            a = -(round(2**j) + round(2**(j-1)) - 2)
            print('Level-4 a =', a)
            b = -(1 - round(2**(j-1))) + 1
            print('Level-4 b =', b)

            for k in range(a, b):
                k_list4.append(k)
                qj[4][k + abs(a)] = -1/256*(dirac(k-7) + 3*dirac(k-6) + 6*dirac(k-5) + 10*dirac(k-4) + 15*dirac(k-3)
                            + 21*dirac(k-2) + 28*dirac(k-1) + 36*dirac(k) + 41*dirac(k+1) + 43*dirac(k+2)
                            + 42*dirac(k+3) + 38*dirac(k+4) + 31*dirac(k+5) + 21*dirac(k+6) + 8*dirac(k+7)
                            - 8*dirac(k+8) - 21*dirac(k+9) - 31*dirac(k+10) - 38*dirac(k+11) - 42*dirac(k+12)
                            - 43*dirac(k+13) - 41*dirac(k+14) - 36*dirac(k+15) - 28*dirac(k+16) - 21*dirac(k+17)
                            - 15*dirac(k+18) - 10*dirac(k+19) - 6*dirac(k+20) - 3*dirac(k+21) - dirac(k+22))

            impulse_response_4 = qj[4][:len(k_list4)]
            detail_coeffs_4 = np.convolve(x, impulse_response_4, mode='same')            

            # ================================
            # DWT Level-5 (manual filter bank)
            # ================================
            k_list5 = []
            j = 5
            a = -(round(2**j) + round(2**(j-1)) - 2)
            print('Level-5 a =', a)
            b = -(1 - round(2**(j-1))) + 1
            print('Level-5 b =', b)

            for k in range(a, b):
                k_list5.append(k)
                qj[5][k + abs(a)] = -1/(512)*(dirac(k-15) + 3*dirac(k-14) + 6*dirac(k-13) + 10*dirac(k-12) + 15*dirac(k-11) + 21*dirac(k-10)
                            + 28*dirac(k-9) + 36*dirac(k-8) + 45*dirac(k-7) + 55*dirac(k-6) + 66*dirac(k-5) + 78*dirac(k-4)
                            + 91*dirac(k-3) + 105*dirac(k-2) + 120*dirac(k-1) + 136*dirac(k) + 149*dirac(k+1) + 159*dirac(k+2)
                            + 166*dirac(k+3) + 170*dirac(k+4) + 171*dirac(k+5) + 169*dirac(k+6) + 164*dirac(k+7) + 156*dirac(k+8)
                            + 145*dirac(k+9) + 131*dirac(k+10) + 114*dirac(k+11) + 94*dirac(k+12) + 71*dirac(k+13) + 45*dirac(k+14)
                            + 16*dirac(k+15) - 16*dirac(k+16) - 45*dirac(k+17) - 71*dirac(k+18) - 94*dirac(k+19) - 114*dirac(k+20)
                            - 131*dirac(k+21) - 145*dirac(k+22) - 156*dirac(k+23) - 164*dirac(k+24) - 169*dirac(k+25)
                            - 171*dirac(k+26) - 170*dirac(k+27) - 166*dirac(k+28) - 159*dirac(k+29) - 149*dirac(k+30)
                            - 136*dirac(k+31) - 120*dirac(k+32) - 105*dirac(k+33) - 91*dirac(k+34) - 78*dirac(k+35)
                            - 66*dirac(k+36) - 55*dirac(k+37) - 45*dirac(k+38) - 36*dirac(k+39) - 28*dirac(k+40)
                            - 21*dirac(k+41) - 15*dirac(k+42) - 10*dirac(k+43) - 6*dirac(k+44) - 3*dirac(k+45)
                            - dirac(k+46))

            impulse_response_5 = qj[5][:len(k_list5)]
            detail_coeffs_5 = np.convolve(x, impulse_response_5, mode='same')

            # ================================
            # DWT Level-6 (manual filter bank)
            # ================================
            k_list6 = []
            j = 6
            a = -(round(2**j) + round(2**(j-1)) - 2)
            print('Level-6 a =', a)
            b = -(1 - round(2**(j-1))) + 1
            print('Level-6 b =', b)

            for k in range(a, b):
                k_list6.append(k)
                qj[6][k + abs(a)] = -1 / 1024 * (dirac(k - 31) + 3 * dirac(k - 30) + 6 * dirac(k - 29) + 10 * dirac(k - 28) +
                        15 * dirac(k - 27) + 21 * dirac(k - 26) + 28 * dirac(k - 25) + 36 * dirac(k - 24) +
                        45 * dirac(k - 23) + 55 * dirac(k - 22) + 66 * dirac(k - 21) + 78 * dirac(k - 20) +
                        91 * dirac(k - 19) + 105 * dirac(k - 18) + 120 * dirac(k - 17) + 136 * dirac(k - 16) +
                        153 * dirac(k - 15) + 171 * dirac(k - 14) + 190 * dirac(k - 13) + 210 * dirac(k - 12) +
                        231 * dirac(k - 11) + 253 * dirac(k - 10) + 276 * dirac(k - 9) + 300 * dirac(k - 8) +
                        325 * dirac(k - 7) + 351 * dirac(k - 6) + 378 * dirac(k - 5) + 406 * dirac(k - 4) +
                        435 * dirac(k - 3) + 465 * dirac(k - 2) + 496 * dirac(k - 1) + 528 * dirac(k) +
                        557 * dirac(k + 1) + 583 * dirac(k + 2) + 606 * dirac(k + 3) + 626 * dirac(k + 4) +
                        643 * dirac(k + 5) + 657 * dirac(k + 6) + 668 * dirac(k + 7) + 676 * dirac(k + 8) +
                        681 * dirac(k + 9) + 683 * dirac(k + 10) + 682 * dirac(k + 11) + 678 * dirac(k + 12) +
                        671 * dirac(k + 13) + 661 * dirac(k + 14) + 648 * dirac(k + 15) + 632 * dirac(k + 16) +
                        613 * dirac(k + 17) + 591 * dirac(k + 18) + 566 * dirac(k + 19) + 538 * dirac(k + 20) +
                        507 * dirac(k + 21) + 473 * dirac(k + 22) + 436 * dirac(k + 23) + 396 * dirac(k + 24) +
                        353 * dirac(k + 25) + 307 * dirac(k + 26) + 258 * dirac(k + 27) + 206 * dirac(k + 28) +
                        151 * dirac(k + 29) + 93 * dirac(k + 30) + 32 * dirac(k + 31) - 32 * dirac(k + 32) -
                        93 * dirac(k + 33) - 151 * dirac(k + 34) - 206 * dirac(k + 35) - 258 * dirac(k + 36) -
                        307 * dirac(k + 37) - 353 * dirac(k + 38) - 396 * dirac(k + 39) - 436 * dirac(k + 40) -
                        473 * dirac(k + 41) - 507 * dirac(k + 42) - 538 * dirac(k + 43) - 566 * dirac(k + 44) -
                        591 * dirac(k + 45) - 613 * dirac(k + 46) - 632 * dirac(k + 47) - 648 * dirac(k + 48) -
                        661 * dirac(k + 49) - 671 * dirac(k + 50) - 678 * dirac(k + 51) - 682 * dirac(k + 52) -
                        683 * dirac(k + 53) - 681 * dirac(k + 54) - 676 * dirac(k + 55) - 668 * dirac(k + 56) -
                        657 * dirac(k + 57) - 643 * dirac(k + 58) - 626 * dirac(k + 59) - 606 * dirac(k + 60) -
                        583 * dirac(k + 61) - 557 * dirac(k + 62) - 528 * dirac(k + 63) - 496 * dirac(k + 64) -
                        465 * dirac(k + 65) - 435 * dirac(k + 66) - 406 * dirac(k + 67) - 378 * dirac(k + 68) -
                        351 * dirac(k + 69) - 325 * dirac(k + 70) - 300 * dirac(k + 71) - 276 * dirac(k + 72) -
                        253 * dirac(k + 73) - 231 * dirac(k + 74) - 210 * dirac(k + 75) - 190 * dirac(k + 76) -
                        171 * dirac(k + 77) - 153 * dirac(k + 78) - 136 * dirac(k + 79) - 120 * dirac(k + 80) -
                        105 * dirac(k + 81) - 91 * dirac(k + 82) - 78 * dirac(k + 83) - 66 * dirac(k + 84) -
                        55 * dirac(k + 85) - 45 * dirac(k + 86) - 36 * dirac(k + 87) - 28 * dirac(k + 88) -
                        21 * dirac(k + 89) - 15 * dirac(k + 90) - 10 * dirac(k + 91) - 6 * dirac(k + 92) -
                        3 * dirac(k + 93) - dirac(k + 94))

            impulse_response_6 = qj[6][:len(k_list6)]
            detail_coeffs_6 = np.convolve(x, impulse_response_6, mode='same')

            # ================================
            # DWT Level-7 (manual filter bank)
            # ================================
            k_list7 = []
            j = 7
            a = -(round(2**j) + round(2**(j-1)) - 2)
            print('Level-7 a =', a)
            b = -(1 - round(2**(j-1))) + 1
            print('Level-7 b =', b)

            for k in range(a, b):
                k_list7.append(k)
                qj[7][k + abs(a)] = -1 / 131072 * (
                dirac(k-63) + 3*dirac(k-62) + 6*dirac(k-61) + 10*dirac(k-60) + 15*dirac(k-59) +
                21*dirac(k-58) + 28*dirac(k-57) + 36*dirac(k-56) + 45*dirac(k-55) +
                55*dirac(k-54) + 66*dirac(k-53) + 78*dirac(k-52) + 91*dirac(k-51) +
                105*dirac(k-50) + 120*dirac(k-49) + 136*dirac(k-48) + 153*dirac(k-47) +
                171*dirac(k-46) + 190*dirac(k-45) + 210*dirac(k-44) + 231*dirac(k-43) +
                253*dirac(k-42) + 276*dirac(k-41) + 300*dirac(k-40) + 325*dirac(k-39) +
                351*dirac(k-38) + 378*dirac(k-37) + 406*dirac(k-36) + 435*dirac(k-35) +
                465*dirac(k-34) + 496*dirac(k-33) + 528*dirac(k-32) + 561*dirac(k-31) +
                595*dirac(k-30) + 630*dirac(k-29) + 666*dirac(k-28) + 703*dirac(k-27) +
                741*dirac(k-26) + 780*dirac(k-25) + 820*dirac(k-24) + 861*dirac(k-23) +
                903*dirac(k-22) + 946*dirac(k-21) + 990*dirac(k-20) + 1035*dirac(k-19) +
                1081*dirac(k-18) + 1128*dirac(k-17) + 1176*dirac(k-16) + 1225*dirac(k-15) +
                1275*dirac(k-14) + 1326*dirac(k-13) + 1378*dirac(k-12) + 1431*dirac(k-11) +
                1485*dirac(k-10) + 1540*dirac(k-9) + 1596*dirac(k-8) + 1653*dirac(k-7) +
                1711*dirac(k-6) + 1770*dirac(k-5) + 1830*dirac(k-4) + 1891*dirac(k-3) +
                1953*dirac(k-2) + 2016*dirac(k-1) + 2080*dirac(k) + 2141*dirac(k+1) +
                2199*dirac(k+2) + 2254*dirac(k+3) + 2306*dirac(k+4) + 2355*dirac(k+5) +
                2401*dirac(k+6) + 2444*dirac(k+7) + 2484*dirac(k+8) + 2521*dirac(k+9) +
                2555*dirac(k+10) + 2586*dirac(k+11) + 2614*dirac(k+12) + 2639*dirac(k+13) +
                2661*dirac(k+14) + 2680*dirac(k+15) + 2696*dirac(k+16) + 2709*dirac(k+17) +
                2719*dirac(k+18) + 2726*dirac(k+19) + 2730*dirac(k+20) + 2731*dirac(k+21) +
                2729*dirac(k+22) + 2724*dirac(k+23) + 2716*dirac(k+24) + 2705*dirac(k+25) +
                2691*dirac(k+26) + 2674*dirac(k+27) + 2654*dirac(k+28) + 2631*dirac(k+29) +
                2605*dirac(k+30) + 2576*dirac(k+31) + 2544*dirac(k+32) + 2509*dirac(k+33) +
                2471*dirac(k+34) + 2430*dirac(k+35) + 2386*dirac(k+36) + 2339*dirac(k+37) +
                2289*dirac(k+38) + 2236*dirac(k+39) + 2180*dirac(k+40) + 2121*dirac(k+41) +
                2059*dirac(k+42) + 1994*dirac(k+43) + 1926*dirac(k+44) + 1855*dirac(k+45) +
                1781*dirac(k+46) + 1704*dirac(k+47) + 1624*dirac(k+48) + 1541*dirac(k+49) +
                1455*dirac(k+50) + 1366*dirac(k+51) + 1274*dirac(k+52) + 1179*dirac(k+53) +
                1081*dirac(k+54) + 980*dirac(k+55) + 876*dirac(k+56) + 769*dirac(k+57) +
                659*dirac(k+58) + 546*dirac(k+59) + 430*dirac(k+60) + 311*dirac(k+61) +
                189*dirac(k+62) + 64*dirac(k+63) - 64*dirac(k+64) - 189*dirac(k+65) -
                311*dirac(k+66) - 430*dirac(k+67) - 546*dirac(k+68) - 659*dirac(k+69) -
                769*dirac(k+70) - 876*dirac(k+71) - 980*dirac(k+72) - 1081*dirac(k+73) -
                1179*dirac(k+74) - 1274*dirac(k+75) - 1366*dirac(k+76) - 1455*dirac(k+77) -
                1541*dirac(k+78) - 1624*dirac(k+79) - 1704*dirac(k+80) - 1781*dirac(k+81) -
                1855*dirac(k+82) - 1926*dirac(k+83) - 1994*dirac(k+84) - 2059*dirac(k+85) -
                2121*dirac(k+86) - 2180*dirac(k+87) - 2236*dirac(k+88) - 2289*dirac(k+89) -
                2339*dirac(k+90) - 2386*dirac(k+91) - 2430*dirac(k+92) - 2471*dirac(k+93) -
                2509*dirac(k+94) - 2544*dirac(k+95) - 2576*dirac(k+96) - 2605*dirac(k+97) -
                2631*dirac(k+98) - 2654*dirac(k+99) - 2674*dirac(k+100) - 2691*dirac(k+101) -
                2705*dirac(k+102) - 2716*dirac(k+103) - 2724*dirac(k+104) - 2729*dirac(k+105) -
                2731*dirac(k+106) - 2730*dirac(k+107) - 2726*dirac(k+108) - 2719*dirac(k+109) -
                2709*dirac(k+110) - 2696*dirac(k+111) - 2680*dirac(k+112) - 2661*dirac(k+113) -
                2639*dirac(k+114) - 2614*dirac(k+115) - 2586*dirac(k+116) - 2555*dirac(k+117) -
                2521*dirac(k+118) - 2484*dirac(k+119) - 2444*dirac(k+120) - 2401*dirac(k+121) -
                2355*dirac(k+122) - 2306*dirac(k+123) - 2254*dirac(k+124) - 2199*dirac(k+125) -
                2141*dirac(k+126) - 2080*dirac(k+127) - 2016*dirac(k+128) - 1953*dirac(k+129) -
                1891*dirac(k+130) - 1830*dirac(k+131) - 1770*dirac(k+132) - 1711*dirac(k+133) -
                1653*dirac(k+134) - 1596*dirac(k+135) - 1540*dirac(k+136) - 1485*dirac(k+137) -
                1431*dirac(k+138) - 1378*dirac(k+139) - 1326*dirac(k+140) - 1275*dirac(k+141) -
                1225*dirac(k+142) - 1176*dirac(k+143) - 1128*dirac(k+144) - 1081*dirac(k+145) -
                1035*dirac(k+146) - 990*dirac(k+147) - 946*dirac(k+148) - 903*dirac(k+149) -
                861*dirac(k+150) - 820*dirac(k+151) - 780*dirac(k+152) - 741*dirac(k+153) -
                703*dirac(k+154) - 666*dirac(k+155) - 630*dirac(k+156) - 595*dirac(k+157) -
                561*dirac(k+158) - 528*dirac(k+159) - 496*dirac(k+160) - 465*dirac(k+161) -
                435*dirac(k+162) - 406*dirac(k+163) - 378*dirac(k+164) - 351*dirac(k+165) -
                325*dirac(k+166) - 300*dirac(k+167) - 276*dirac(k+168) - 253*dirac(k+169) -
                231*dirac(k+170) - 210*dirac(k+171) - 190*dirac(k+172) - 171*dirac(k+173) -
                153*dirac(k+174) - 136*dirac(k+175) - 120*dirac(k+176) - 105*dirac(k+177) -
                91*dirac(k+178) - 78*dirac(k+179) - 66*dirac(k+180) - 55*dirac(k+181) -
                45*dirac(k+182) - 36*dirac(k+183) - 28*dirac(k+184) - 21*dirac(k+185) -
                15*dirac(k+186) - 10*dirac(k+187) - 6*dirac(k+188) - 3*dirac(k+189) -
                dirac(k+190)
            )

            impulse_response_7 = qj[7][:len(k_list7)]
            detail_coeffs_7 = np.convolve(x, impulse_response_7, mode='same')

            # ================================
            # DWT Level-8 (manual filter bank)
            # ================================
            k_list8 = []
            j = 8
            a = -(round(2**j) + round(2**(j-1)) - 2)
            print('Level-8 a =', a)
            b = -(1 - round(2**(j-1))) + 1
            print('Level-8 b =', b)
            # eq1: dari k-127 sampai k-1
            eq1 = (
                dirac(k-127) + 3*dirac(k-126) + 6*dirac(k-125) + 10*dirac(k-124) + 15*dirac(k-123) +
                21*dirac(k-122) + 28*dirac(k-121) + 36*dirac(k-120) + 45*dirac(k-119) + 55*dirac(k-118) +
                66*dirac(k-117) + 78*dirac(k-116) + 91*dirac(k-115) + 105*dirac(k-114) + 120*dirac(k-113) +
                136*dirac(k-112) + 153*dirac(k-111) + 171*dirac(k-110) + 190*dirac(k-109) + 210*dirac(k-108) +
                231*dirac(k-107) + 253*dirac(k-106) + 276*dirac(k-105) + 300*dirac(k-104) + 325*dirac(k-103) +
                351*dirac(k-102) + 378*dirac(k-101) + 406*dirac(k-100) + 435*dirac(k-99) + 465*dirac(k-98) +
                496*dirac(k-97) + 528*dirac(k-96) + 561*dirac(k-95) + 595*dirac(k-94) + 630*dirac(k-93) +
                666*dirac(k-92) + 703*dirac(k-91) + 741*dirac(k-90) + 780*dirac(k-89) + 820*dirac(k-88) +
                861*dirac(k-87) + 903*dirac(k-86) + 946*dirac(k-85) + 990*dirac(k-84) + 1035*dirac(k-83) +
                1081*dirac(k-82) + 1128*dirac(k-81) + 1176*dirac(k-80) + 1225*dirac(k-79) + 1275*dirac(k-78) +
                1326*dirac(k-77) + 1378*dirac(k-76) + 1431*dirac(k-75) + 1485*dirac(k-74) + 1540*dirac(k-73) +
                1596*dirac(k-72) + 1653*dirac(k-71) + 1711*dirac(k-70) + 1770*dirac(k-69) + 1830*dirac(k-68) +
                1891*dirac(k-67) + 1953*dirac(k-66) + 2016*dirac(k-65) + 2080*dirac(k-64) + 2145*dirac(k-63) +
                2211*dirac(k-62) + 2278*dirac(k-61) + 2346*dirac(k-60) + 2415*dirac(k-59) + 2485*dirac(k-58) +
                2556*dirac(k-57) + 2628*dirac(k-56) + 2701*dirac(k-55) + 2775*dirac(k-54) + 2850*dirac(k-53) +
                2926*dirac(k-52) + 3003*dirac(k-51) + 3081*dirac(k-50) + 3160*dirac(k-49) + 3240*dirac(k-48) +
                3321*dirac(k-47) + 3403*dirac(k-46) + 3486*dirac(k-45) + 3570*dirac(k-44) + 3655*dirac(k-43) +
                3741*dirac(k-42) + 3828*dirac(k-41) + 3916*dirac(k-40) + 4005*dirac(k-39) + 4095*dirac(k-38) +
                4186*dirac(k-37) + 4278*dirac(k-36) + 4371*dirac(k-35) + 4465*dirac(k-34) + 4560*dirac(k-33) +
                4656*dirac(k-32) + 4753*dirac(k-31) + 4851*dirac(k-30) + 4950*dirac(k-29) + 5050*dirac(k-28) +
                5151*dirac(k-27) + 5253*dirac(k-26) + 5356*dirac(k-25) + 5460*dirac(k-24) + 5565*dirac(k-23) +
                5671*dirac(k-22) + 5778*dirac(k-21) + 5886*dirac(k-20) + 5995*dirac(k-19) + 6105*dirac(k-18) +
                6216*dirac(k-17) + 6328*dirac(k-16) + 6441*dirac(k-15) + 6555*dirac(k-14) + 6670*dirac(k-13) +
                6786*dirac(k-12) + 6903*dirac(k-11) + 7021*dirac(k-10) + 7140*dirac(k-9) + 7260*dirac(k-8) +
                7381*dirac(k-7) + 7503*dirac(k-6) + 7626*dirac(k-5) + 7750*dirac(k-4) + 7875*dirac(k-3) +
                8001*dirac(k-2) + 8128*dirac(k-1)
            )

            # eq2: dari k sampai k+127
            eq2 = (
                8256*dirac(k) + 8128*dirac(k+1) + 8001*dirac(k+2) +
                7875*dirac(k+3) + 7750*dirac(k+4) + 7626*dirac(k+5) + 7503*dirac(k+6) + 7381*dirac(k+7) +
                7260*dirac(k+8) + 7140*dirac(k+9) + 7021*dirac(k+10) + 6903*dirac(k+11) + 6786*dirac(k+12) +
                6670*dirac(k+13) + 6555*dirac(k+14) + 6441*dirac(k+15) + 6328*dirac(k+16) + 6216*dirac(k+17) +
                6105*dirac(k+18) + 5995*dirac(k+19) + 5886*dirac(k+20) + 5778*dirac(k+21) + 5671*dirac(k+22) +
                5565*dirac(k+23) + 5460*dirac(k+24) + 5356*dirac(k+25) + 5253*dirac(k+26) + 5151*dirac(k+27) +
                5050*dirac(k+28) + 4950*dirac(k+29) + 4851*dirac(k+30) + 4753*dirac(k+31) + 4656*dirac(k+32) +
                4560*dirac(k+33) + 4465*dirac(k+34) + 4371*dirac(k+35) + 4278*dirac(k+36) + 4186*dirac(k+37) +
                4095*dirac(k+38) + 4005*dirac(k+39) + 3916*dirac(k+40) + 3828*dirac(k+41) + 3741*dirac(k+42) +
                3655*dirac(k+43) + 3570*dirac(k+44) + 3486*dirac(k+45) + 3403*dirac(k+46) + 3321*dirac(k+47) +
                3240*dirac(k+48) + 3160*dirac(k+49) + 3081*dirac(k+50) + 3003*dirac(k+51) + 2926*dirac(k+52) +
                2850*dirac(k+53) + 2775*dirac(k+54) + 2701*dirac(k+55) + 2628*dirac(k+56) + 2556*dirac(k+57) +
                2485*dirac(k+58) + 2415*dirac(k+59) + 2346*dirac(k+60) + 2278*dirac(k+61) + 2211*dirac(k+62) +
                2145*dirac(k+63) + 2080*dirac(k+64) + 2016*dirac(k+65) + 1953*dirac(k+66) + 1891*dirac(k+67) +
                1830*dirac(k+68) + 1770*dirac(k+69) + 1711*dirac(k+70) + 1653*dirac(k+71) + 1596*dirac(k+72) +
                1540*dirac(k+73) + 1485*dirac(k+74) + 1431*dirac(k+75) + 1378*dirac(k+76) + 1326*dirac(k+77) +
                1275*dirac(k+78) + 1225*dirac(k+79) + 1176*dirac(k+80) + 1128*dirac(k+81) + 1081*dirac(k+82) +
                1035*dirac(k+83) + 990*dirac(k+84) + 946*dirac(k+85) + 903*dirac(k+86) + 861*dirac(k+87) +
                820*dirac(k+88) + 780*dirac(k+89) + 741*dirac(k+90) + 703*dirac(k+91) + 666*dirac(k+92) +
                630*dirac(k+93) + 595*dirac(k+94) + 561*dirac(k+95) + 528*dirac(k+96) + 496*dirac(k+97) +
                465*dirac(k+98) + 435*dirac(k+99) + 406*dirac(k+100) + 378*dirac(k+101) + 351*dirac(k+102) +
                325*dirac(k+103) + 300*dirac(k+104) + 276*dirac(k+105) + 253*dirac(k+106) + 231*dirac(k+107) +
                210*dirac(k+108) + 190*dirac(k+109) + 171*dirac(k+110) + 153*dirac(k+111) + 136*dirac(k+112) +
                120*dirac(k+113) + 105*dirac(k+114) + 91*dirac(k+115) + 78*dirac(k+116) + 66*dirac(k+117) +
                55*dirac(k+118) + 45*dirac(k+119) + 36*dirac(k+120) + 28*dirac(k+121) + 21*dirac(k+122) +
                15*dirac(k+123) + 10*dirac(k+124) + 6*dirac(k+125) + 3*dirac(k+126) + dirac(k+127)
            )

            # Persamaan keseluruhan:
            for k in range(a, b):
                k_list8.append(k)
                qj[8][k + abs(a)] = -1 / 1048576 * (eq1 + eq2)
                impulse_response_8 = qj[8][:len(k_list8)]
                detail_coeffs_8 = np.convolve(x, impulse_response_8, mode='same')


            # Definisikan signals dan colors
            signals = [detail_coeffs_1, detail_coeffs_2, detail_coeffs_3, detail_coeffs_4, detail_coeffs_5, detail_coeffs_6, detail_coeffs_7, detail_coeffs_8]
            colors = ['salmon', 'orange', 'green', 'skyblue', 'blue', 'magenta', 'purple', 'darkviolet']

            # Loop untuk buat 8 figure terpisah
            for i in range(8):
                # Buat figure kosong untuk setiap level
                fig_dwt = go.Figure()

                # Tambahkan trace untuk level ini
                fig_dwt.add_trace(
                    go.Scatter(
                        x=waktu,
                        y=signals[i],
                        mode='lines',
                        name=f'DWT Level-{i+1}',
                        line=dict(color=colors[i])
                    )
                )

                # Update layout untuk figure ini
                fig_dwt.update_layout(
                    title=f'DWT Level-{i+1}',
                    xaxis_title='Waktu (s)',
                    yaxis_title='Amplitudo',
                    dragmode='zoom',
                    height=400,  # Ukuran lebih kecil biar nyaman
                )

                # Tampilkan figure di Streamlit
                st.plotly_chart(fig_dwt, use_container_width=True)

  
            
        with tab3:
            st.header("Langkah Pemrosesan")

            # --- Squaring (mengambil nilai absolut) ---
            def squaring(signal):
                squared = np.zeros(np.size(signal))
                for i in range(len(squared)):
                    squared[i] = signal[i] ** 2
                return squared

            # Apply squaring function untuk D3
            squared_signal3 = squaring(detail_coeffs_3)

            # Plot Squaring D3
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=waktu, y=squared_signal3, mode='lines', name='Squared D3', line=dict(color='green')))
            fig1.update_layout(title='Squaring ECG Signal D3', xaxis_title='Waktu (s)', yaxis_title='Amplitudo', dragmode='zoom', height=400)
            st.plotly_chart(fig1, use_container_width=True)

            # --- Moving Average (MAV) ---
            def moving_average(signal, M):
                N = len(signal)
                y1_mav = np.zeros(N)
                y2_mav = np.zeros(N)
                # Forward MA filtering
                for i in range(N):
                    sum_value = 0
                    count = 0
                    for j in range(M):
                        if i - j >= 0:
                            sum_value += signal[i - j]
                            count += 1
                    y1_mav[i] = sum_value / count
                # Backward MA filtering
                for i in range(N):
                    sum_value = 0
                    count = 0
                    for j in range(M):
                        if i + j < N:
                            sum_value += y1_mav[i + j]
                            count += 1
                    y2_mav[i] = sum_value / count
                return y2_mav

            # Apply Moving Average untuk D3
            mav_signal3 = moving_average(squared_signal3, 5)

            # Plot Moving Average D3
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=waktu, y=mav_signal3, mode='lines', name='MAV D3', line=dict(color='green')))
            fig2.update_layout(title='MAV ECG Signal D3', xaxis_title='Waktu (s)', yaxis_title='Amplitudo', dragmode='zoom', height=400)
            st.plotly_chart(fig2, use_container_width=True)

            # --- Thresholding ---
            def threshold(signal, threshold_value, threshold_above):
                threshold_signal = np.zeros(np.size(signal))
                for i in range(len(signal)):
                    if signal[i] > threshold_value:
                        threshold_signal[i] = threshold_above
                    else:
                        threshold_signal[i] = 0
                return threshold_signal

            # Apply Threshold untuk D3
            threshold_signal3 = threshold(mav_signal3, 0.25, 1)

            # Plot Threshold D3
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=waktu, y=threshold_signal3, mode='lines', name='Thresholded D3', line=dict(color='green')))
            fig3.update_layout(title='Thresholding ECG Signal D3', xaxis_title='Waktu (s)', yaxis_title='Amplitudo', dragmode='zoom', height=400)
            st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            st.header("Deteksi Denyut Jantung")

            # Asumsi frekuensi sampling ECG
            fs = 125

            # List untuk menyimpan hasil deteksi puncak dan BPM
            all_peaks = []
            all_bpm = []

            # Siapkan sinyal detail coefficients dan threshold hasil squaring
            detail_signals = [detail_coeffs_1, detail_coeffs_2, detail_coeffs_3]
            thresholded_signals = [threshold_signal3]

            # --- Deteksi puncak R dan hitung BPM untuk setiap sinyal ---
            for i, signal in enumerate(thresholded_signals):
                signal = np.array(signal).flatten()

                # Deteksi puncak R
                peaks, _ = scipy.signal.find_peaks(signal, distance=fs*0.3)
                all_peaks.append(peaks)

                # Hitung RR intervals
                rr_intervals = np.diff(peaks) / fs  # dalam detik

                # Hitung BPM
                bpm_list = []
                if len(rr_intervals) > 0:
                    for rr in rr_intervals:
                        bpm = 60 / rr
                        bpm_list.append(bpm)
                    all_bpm.append(bpm_list)
                else:
                    all_bpm.append([])  # Jika tidak ada puncak R terdeteksi

            # --- Fokus tampilkan sinyal D3 dengan R-peak deteksi ---
            st.subheader("Sinyal D3 dengan Deteksi R-peak")

            fig_d3 = go.Figure()

                # Plot detail coefficients D3
            fig_d3.add_trace(go.Scatter(
                    x=waktu,
                    y=detail_signals[2],
                    mode='lines',
                    name='Detail Coeffs D3',
                    line=dict(color='grey')
                ))

                # Plot thresholded signal D3
            fig_d3.add_trace(go.Scatter(
                    x=waktu,
                    y=thresholded_signals[0],
                    mode='lines',
                    name='Thresholded Signal D3',
                    line=dict(color='green')
                ))

                # Plot detected R-peaks
            if len(all_peaks[0]) > 0:
                    fig_d3.add_trace(go.Scatter(
                        x=np.array(waktu)[all_peaks[0]],
                        y=thresholded_signals[0][all_peaks[0]],
                        mode='markers',
                        name='Detected R-peaks',
                        marker=dict(color='red', size=8, symbol='circle')
                    ))
            else:
                    st.write("Tidak ada puncak R terdeteksi pada sinyal D3.")

            fig_d3.update_layout(
                    title='Deteksi R-peak pada Sinyal D3',
                    xaxis_title='Waktu (s)',
                    yaxis_title='Amplitudo',
                    dragmode='zoom',
                    height=500
                )

            st.plotly_chart(fig_d3, use_container_width=True)

            # --- Plot HRV (RR Interval) untuk D3 ---
            st.subheader("Plot HRV (RR Interval) untuk D3")

            if len(all_peaks[0]) > 1:
                rr_intervals_d3 = np.diff(all_peaks[0]) / fs  # RR interval dalam detik
                beat_numbers = np.arange(1, len(rr_intervals_d3) + 1)

                fig_hrv, ax_hrv = plt.subplots(figsize=(12, 6))
                ax_hrv.plot(beat_numbers, rr_intervals_d3, marker='o', linestyle='-', color='blue')
                ax_hrv.set_xlabel('Beat Number')
                ax_hrv.set_ylabel('RR Interval (s)')
                ax_hrv.set_title('HRV Plot for D3')
                ax_hrv.grid(True)
                st.pyplot(fig_hrv)
            else:
                st.write("Tidak cukup puncak R terdeteksi untuk menghitung HRV.")

            # --- Tampilkan hasil BPM rata-rata khusus D3 ---
            st.subheader("Hasil BPM D3")
            if len(all_bpm[0]) > 0:
                avg_bpm = np.mean(all_bpm[0])
                st.write(f"Rata-rata BPM Sinyal D3 = {avg_bpm:.2f}")
            else:
                st.write("Tidak cukup puncak R terdeteksi pada sinyal D3.")


        with tab5:
            st.header("Perbandingan Sinyal Pernapasan dan DWT Level-7")

            # Scale DWT Level-7 signal dulu
            dwt7 = detail_coeffs_7 * 5 + 0.5

            # Buat figure Plotly
            fig5 = go.Figure()

            # Respiratory Signal
            fig5.add_trace(go.Scatter(
                x=waktu,
                y=resp,
                mode='lines',
                name='Respiratory Signal',
                line=dict(color='pink')
            ))

            # DWT Level-7 (Scaled ECG)
            fig5.add_trace(go.Scatter(
                x=waktu,
                y=dwt7,
                mode='lines',
                name='DWT Level-7 (Scaled ECG)',
                line=dict(color='blue')
            ))

            # Layout pengaturan
            fig5.update_layout(
                title='Respiratory Signal vs DWT Level-7 ECG',
                xaxis_title='Waktu (s)',
                yaxis_title='Amplitudo',
                dragmode='zoom',
                height=500,
                yaxis=dict(range=[-1, 1.5]),
                legend=dict(x=0, y=1)
            )

            # Tampilkan di Streamlit
            st.plotly_chart(fig5, use_container_width=True)

        with tab6:
            st.header("Time Domain Analysis")

            # Set pandas biar nampilin semua baris dan kolom
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.expand_frame_repr', False)

            # --- Hitung RR Intervals ---
            rr_intervals_all = []
            for i in range(3):
                rr_intervals = np.diff(all_peaks[0]) / fs
                rr_intervals_all.append(rr_intervals)

            # --- Tampilkan hanya sinyal ketiga ---
            rr_intervals = rr_intervals_all[0]

            # SDNN
            rr_intervals_d3 = rr_intervals_all[0]
            if len(rr_intervals_d3) > 1:
                mean_rr = np.mean(rr_intervals_d3)
                squared_diffs = (rr_intervals_d3 - mean_rr) ** 2
                sum_squared_diffs = np.sum(squared_diffs)
                variance = sum_squared_diffs / (len(rr_intervals_d3) - 1)
                sdnn_d3 = np.sqrt(variance)
                sdnn_d3_ms = sdnn_d3 * 1000
                st.write(f"**SDNN untuk Sinyal D3:** {sdnn_d3_ms:.4f} ms")
            else:
                st.write("Tidak cukup RR interval untuk menghitung SDNN pada Sinyal D3.")

            # RMSSD
            if len(rr_intervals_d3) > 1:
                successive_diffs = np.diff(rr_intervals_d3)
                squared_successive_diffs = successive_diffs ** 2
                sum_squared_successive_diffs = np.sum(squared_successive_diffs)
                mean_squared_successive_diffs = sum_squared_successive_diffs / (len(rr_intervals_d3) - 1)
                rmssd_d3 = np.sqrt(mean_squared_successive_diffs)
                rmssd_d3_ms = rmssd_d3 * 1000
                st.write(f"**RMSSD untuk Sinyal D3:** {rmssd_d3_ms:.4f} ms")
            else:
                st.write("Tidak cukup RR interval untuk menghitung RMSSD pada Sinyal D3.")

            # pNN50
            def calculate_pnn50(rr_intervals):
                rr_intervals_ms = rr_intervals * 1000
                rr_diff = np.abs(np.diff(rr_intervals_ms))
                count_diff_greater_than_50 = np.sum(rr_diff > 50)
                total_rr_intervals = len(rr_intervals_ms) - 1
                if total_rr_intervals > 0:
                    pnn50 = (count_diff_greater_than_50 / total_rr_intervals) * 100
                else:
                    pnn50 = 0.0
                return pnn50

            pnn50_value_d3 = calculate_pnn50(rr_intervals_d3)
            st.write(f"**pNN50 untuk Sinyal D3:** {pnn50_value_d3:.2f}%")

            # SDSD
            def calculate_sdsd(rr_intervals):
                rr_intervals = np.array(rr_intervals)
                delta_rr = np.diff(rr_intervals)
                N = len(rr_intervals)
                if N < 3:
                    return 0.0
                mean_delta_rr = np.sum(delta_rr) / (N - 1)
                sdsd_numerator = np.sum((delta_rr - mean_delta_rr) ** 2)
                sdsd = np.sqrt(sdsd_numerator / (N - 2))
                sdsd_ms = sdsd * 1000
                return sdsd_ms

            sdsd_value = calculate_sdsd(rr_intervals_d3)
            st.write(f"**SDSD untuk Sinyal D3:** {sdsd_value:.4f} ms")

            # Plot Histogram RR Intervals
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(rr_intervals_d3, bins=30, color='magenta', edgecolor='black')
            ax.set_xlabel('RR Interval (seconds)')
            ax.set_ylabel('Frequency')
            ax.set_title('Histogram RR Intervals')
            ax.grid(axis='y')
            st.pyplot(fig)


        with tab7 :
            st.header("Frequency Domain Analysis")

            # --- Interpolasi ke 4 Hz (sampling uniform) ---
            fs_interp = 4
            duration = np.cumsum(rr_intervals)
            t_uniform = np.arange(0, duration[-1], 1 / fs_interp)
            rr_interp = np.interp(t_uniform, duration, rr_intervals)

            # --- Parameter Welch Manual ---
            window_size = 240         # 60 detik @ 4 Hz
            overlap = 120             # 50% overlap
            step = window_size - overlap
            pad_to = 512              # zero-padding

            # --- Fungsi Hamming Window Manual ---
            def hamming(N):
                return [0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1)) for n in range(N)]

            # --- DFT Manual (bukan FFT) ---
            def dft(signal, fs):
                N = len(signal)
                psd = []
                freqs = []
                for k in range(N // 2 + 1):
                    real = sum(signal[n] * math.cos(2 * math.pi * k * n / N) for n in range(N))
                    imag = sum(-signal[n] * math.sin(2 * math.pi * k * n / N) for n in range(N))
                    power = (real**2 + imag**2) / N
                    psd.append(power)
                    freqs.append(k * fs / N)
                return np.array(freqs), np.array(psd)

            # --- Proses Welch (Sum of PSD Segmen) ---
            segments = []
            for start in range(0, len(rr_interp) - window_size + 1, step):
                segment = rr_interp[start:start + window_size]
                segment = segment - np.mean(segment)  # detrend
                window = hamming(window_size)
                windowed = [s * w for s, w in zip(segment, window)]
                padded = windowed + [0.0] * (pad_to - window_size)
                freqs, psd = dft(padded, fs_interp)
                segments.append(psd)

            # --- Jumlahkan Semua Segmen PSD (bukan rata-rata) ---
            psd_sum = np.sum(segments, axis=0)

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(freqs, np.array(psd_sum) * 1000, color='black', linewidth=1.5)
            ax.set_title("Welch Periodogram")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power (×10⁻³ s²/Hz)")
            ax.set_xlim(0, 0.5)
            ax.grid(True)

            # --- Mewarnai area sesuai rentang frekuensi dan data PSD ---
            ax.fill_between(freqs, 0, np.array(psd_sum)*1000, where=(freqs >= 0.003) & (freqs < 0.04), color='blue', alpha=0.5, label='VLF')
            ax.fill_between(freqs, 0, np.array(psd_sum)*1000, where=(freqs >= 0.04) & (freqs < 0.15), color='purple', alpha=0.5, label='LF')
            ax.fill_between(freqs, 0, np.array(psd_sum)*1000, where=(freqs >= 0.15) & (freqs < 0.4), color='cyan', alpha=0.5, label='HF')
            ax.legend(loc='upper right', frameon=False)
            plt.tight_layout()
            st.pyplot(fig)

            # --- Hitung Band Power ---
            def band_power(freqs, psd, f_low, f_high):
                return np.sum(psd[(freqs >= f_low) & (freqs < f_high)])

            vlf_power = band_power(freqs, psd_sum, 0.003, 0.04)
            lf_power  = band_power(freqs, psd_sum, 0.04, 0.15)
            hf_power  = band_power(freqs, psd_sum, 0.15, 0.4)
            total_power = vlf_power + lf_power + hf_power

            # --- Hitung Persentase (normal unit) ---
            lf_nu = lf_power / (total_power - vlf_power) if (total_power - vlf_power) > 0 else 0
            hf_nu = hf_power / (total_power - vlf_power) if (total_power - vlf_power) > 0 else 0

            # --- Tampilkan di GUI Streamlit ---
            st.subheader("Band Power (ms²)")
            st.write(f"VLF: {vlf_power*1000:.1f} ms²")
            st.write(f"LF:  {lf_power*1000:.1f} ms² ({lf_nu*100:.1f}%)")
            st.write(f"HF:  {hf_power*1000:.1f} ms² ({hf_nu*100:.1f}%)")
            st.write(f"Total Power: {total_power*1000:.1f} ms²")

            # --- LF/HF Bar Chart ---
            labels = ['LF', 'HF']
            values = [lf_power*1000, hf_power*1000]  # dikali 1000 agar dalam ms²
            colors = ['purple', 'cyan']
            lf_hf_ratio = lf_power / hf_power if hf_power != 0 else float('inf')

            fig2, ax2 = plt.subplots(figsize=(3, 3))
            ax2.bar(labels, values, color=colors)
            ax2.set_ylabel("Power (ms²)")
            ax2.set_title("LF and HF Power")
            ax2.text(0.5, max(values) * 1.05, f"LF/HF Ratio = {lf_hf_ratio:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax2.set_ylim(0, max(values) * 1.2)
            plt.tight_layout()
            st.pyplot(fig2)

            # --- Autonomic Balance Grid ---
            def scale_to_grid(value, thresholds=(0.33, 0.66)):
                if value < thresholds[0]:
                    return 0  # Low
                elif value < thresholds[1]:
                    return 1  # Medium
                else:
                    return 2  # High

            x = scale_to_grid(lf_nu)
            y = scale_to_grid(hf_nu)
            colors_grid = [
                ['#FF9999', '#FFCC66', '#FF9999'],
                ['#FFFF66', '#66CC66', '#FFFF66'],
                ['#FFFF66', '#99FF99', '#99FF99']
            ]

            fig3, ax3 = plt.subplots(figsize=(5, 5))
            for i in range(3):  # row (HF)
                for j in range(3):  # column (LF)
                    ax3.add_patch(plt.Rectangle((j, i), 1, 1, color=colors_grid[i][j], ec='black'))

            ax3.plot(lf_nu * 3, hf_nu * 3, 'ro', markersize=10)

            ax3.set_xticks([0.5, 1.5, 2.5])
            ax3.set_xticklabels(['Low', 'Medium', 'High'])
            ax3.set_yticks([0.5, 1.5, 2.5])
            ax3.set_yticklabels(['Low', 'Medium', 'High'])
            ax3.set_xlabel('Sympathetic NS - LF')
            ax3.set_ylabel('Parasympathetic NS - HF')
            ax3.set_xlim(0, 3)
            ax3.set_ylim(0, 3)
            ax3.set_title('Autonomic Balance Diagram')
            ax3.set_aspect('equal')
            plt.tight_layout()
            st.pyplot(fig3)


        with tab8 :
            st.header("Linear Analysis")

            import pyhrv.nonlinear as nl

            # Ambil RR interval sinyal ketiga (dalam detik)
            rr_intervals = rr_intervals_all[2]

            # Konversi ke milidetik
            rr_intervals_ms = rr_intervals * 1000

            # Hitung Poincaré plot tanpa langsung menampilkannya
            results = nl.poincare(rr_intervals_ms, show=False)

            # Ambil objek figure dari plot Poincaré
            fig = plt.gcf()

            # Tampilkan plot di Streamlit
            st.pyplot(fig)

            # Tampilkan nilai SD1, SD2, dan rasio
            st.subheader("Hasil Poincaré untuk Sinyal D3:")
            st.write(f"SD1: {results['sd1']:.2f} ms")
            st.write(f"SD2: {results['sd2']:.2f} ms")
            st.write(f"SD1/SD2 Ratio: {results['sd_ratio']:.2f}")



if __name__ == "__main__":
    main()