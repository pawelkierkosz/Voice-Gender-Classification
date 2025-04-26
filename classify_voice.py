import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import hamming

WIELKOSC_OKNA = 2048      # Rozmiar okna (dla FFT)
KROK_NAKLADANIA = WIELKOSC_OKNA // 2
CZYNNIK_HARMONICZNY = 4   # Liczba harmonicznych w redukcji


ZAKRES_MESKI = (85, 180)
ZAKRES_ZENSKI = (165, 255)

def klasyfikacja_hps(probki, fs):
    dlugosc_prob = len(probki)
    if dlugosc_prob < WIELKOSC_OKNA:
        # Zero-padding do rozmiaru WIELKOSC_OKNA, jeśli za krótkie
        tmp = np.zeros(WIELKOSC_OKNA)
        tmp[:dlugosc_prob] = probki
        probki = tmp

    # Przygotowanie okna i wyliczenie, ile segmentów obsłużymy
    maska_okna = hamming(WIELKOSC_OKNA, sym=False)
    ile_segmentow = max(1, (len(probki) - WIELKOSC_OKNA) // KROK_NAKLADANIA + 1)

    suma_m = 0.0
    suma_k = 0.0

    for seg_index in range(ile_segmentow):
        start = seg_index * KROK_NAKLADANIA
        koniec = start + WIELKOSC_OKNA
        wycinek = probki[start:koniec]
        if len(wycinek) < WIELKOSC_OKNA:
            # Uzupełnienie zerami gdy ostatni segment za krótki
            uzupelniony = np.zeros(WIELKOSC_OKNA)
            uzupelniony[:len(wycinek)] = wycinek
            wycinek = uzupelniony

        wycinek_okno = wycinek * maska_okna
        widmo = np.fft.rfft(wycinek_okno, WIELKOSC_OKNA)
        widmo_mocy = np.abs(widmo) ** 2

        # HPS – redukcja harmoniczna
        hps = widmo_mocy.copy()
        for h in range(2, CZYNNIK_HARMONICZNY + 1):
            w_dol = widmo_mocy[::h]
            hps[:len(w_dol)] *= w_dol

        czestotliwosci = np.fft.rfftfreq(WIELKOSC_OKNA, d=1/fs)

        # Sumowanie energii w zakresie męskim i żeńskim
        maska_m = (czestotliwosci >= ZAKRES_MESKI[0]) & (czestotliwosci <= ZAKRES_MESKI[1])
        maska_k = (czestotliwosci >= ZAKRES_ZENSKI[0]) & (czestotliwosci <= ZAKRES_ZENSKI[1])

        suma_m += np.sum(hps[maska_m])
        suma_k += np.sum(hps[maska_k])

    if suma_m > suma_k:
        return 'M'
    return 'K'

def rozpoznaj_plik_nagranie(plik):
    fs, probki = wavfile.read(plik)
    # Jeśli stereo, wybieramy pierwszy kanał
    if len(probki.shape) > 1:
        probki = probki[:, 0]

    maks = np.max(np.abs(probki))
    if maks != 0:
        probki = probki / maks

    wynik = klasyfikacja_hps(probki, fs)
    print(wynik)


def skanuj_wav():
    sciezka_aktualna = os.getcwd()
    lista_wav = [n for n in os.listdir(sciezka_aktualna) if n.lower().endswith('.wav')]

    if not lista_wav:
        print("Brak plików .wav do analizy.")
        return

    poprawne_licznik = 0
    lacznie_plikow = 0

    for nazwa in lista_wav:
        sciezka_do_pliku = os.path.join(sciezka_aktualna, nazwa)

        plec_z_nazwy = nazwa.split('.')[-2][-1].upper()
        if plec_z_nazwy not in ['M', 'K']:
            continue

        fs, probki = wavfile.read(sciezka_do_pliku)
        if len(probki.shape) > 1:
            probki = probki[:, 0]

        mx = np.max(np.abs(probki))
        if mx != 0:
            probki = probki / mx

        przypisane = klasyfikacja_hps(probki, fs)

        print(f"{nazwa}: Oczek. {plec_z_nazwy}, Rozpoznano {przypisane}")

        if przypisane == plec_z_nazwy:
            poprawne_licznik += 1
        lacznie_plikow += 1

    if lacznie_plikow > 0:
        trafnosc = 100.0 * poprawne_licznik / lacznie_plikow
        print(f"\nTrafność: {trafnosc:.2f}%  ({poprawne_licznik}/{lacznie_plikow})")
    else:
        print("Nie rozpoznano żadnych plików (brak nazwy sugerującej płeć).")


def main():
    if len(sys.argv) == 1:
        skanuj_wav()
    else:
        sciezka_pliku = sys.argv[1]
        if not os.path.isfile(sciezka_pliku):
            print(f"Nie znaleziono pliku: {sciezka_pliku}")
            sys.exit(1)
        rozpoznaj_plik_nagranie(sciezka_pliku)

if __name__ == "__main__":
    main()
