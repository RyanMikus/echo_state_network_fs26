"""
pairs_trading.data_io.collect

Datensammlung (Collection) für das pairs_trading Projekt:

- Universe: S&P 500 Symbols via Wikipedia Tabelle.
- Intraday Daten: direkter Abruf über den Yahoo Finance "chart" Endpoint (ohne yfinance).

Wichtige Hinweise:
- Netzwerkzugriffe (Wikipedia, Yahoo) sind Teil des Laufzeitverhaltens.
- Print-Ausgaben (inkl. Progress-Bar in einer Zeile) sind Teil des beobachtbaren Verhaltens.
- collect_sp500_intraday_data schluckt Exceptions pro Symbol bewusst, um einen Durchlauf nicht
  wegen einzelner Fehler abzubrechen. Dadurch sind Datenlücken möglich.
"""

import sys
import time
import datetime as dt

import pandas as pd
import requests


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def get_sp500_universe(replace_dot=True, user_agent="Mozilla/5.0", verbose=True):
    """
    Lädt die Liste der S&P 500 Ticker-Symbole von Wikipedia.

    Parameter:
    - replace_dot: wenn True, werden '.' durch '-' ersetzt (z. B. BRK.B -> BRK-B),
      was gängigerweise mit Yahoo Symbolkonventionen kompatibel ist.
    - user_agent: User-Agent Header für requests.get
    - verbose: wenn True, werden Statusprints ausgegeben

    Rückgabe:
    - Liste von Symbolstrings

    Side effects:
    - Netzwerkzugriff auf Wikipedia.
    - Prints bei verbose=True.
    """
    if verbose:
        print("Lade S&P 500 Universe von Wikipedia …")

    headers = {"User-Agent": user_agent}
    html = requests.get(WIKI_URL, headers=headers, timeout=30).text
    tables = pd.read_html(html)

    df = tables[0]
    symbols = df["Symbol"].astype(str).tolist()

    if replace_dot:
        symbols = [s.replace(".", "-") for s in symbols]

    if verbose:
        print("Universe Größe:", len(symbols))

    return symbols


def date_to_unix_periods(date_str, tz="America/New_York"):
    """
    Konvertiert ein ISO-Datum (YYYY-MM-DD) in (period1, period2) Unix-Timestamps.

    period1/period2 definieren das Zeitfenster für den Yahoo-Chart-Endpunkt.
    Hier wird bewusst ein Tagesfenster [00:00, 24:00) in der angegebenen Zeitzone gewählt.
    """
    d = dt.date.fromisoformat(date_str)
    start = pd.Timestamp(dt.datetime(d.year, d.month, d.day, 0, 0), tz=tz)
    end = start + pd.Timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


def parse_yahoo_chart_json(symbol, payload, tz="America/New_York"):
    """
    Parst die JSON-Antwort des Yahoo-Chart-Endpunkts in einen DataFrame.

    Verhalten:
    - Wenn der Payload einen chart-error enthält: RuntimeError.
    - Wenn keine Result-Liste / keine Timestamps / keine Quote-Indikatoren vorhanden:
      Rückgabe eines leeren DataFrames.
    - 'datetime' wird zu timezone-aware timestamps konvertiert und nach tz konvertiert.
    """
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError("Yahoo chart error for {}: {}".format(symbol, error))

    result_list = chart.get("result") or []
    if not result_list:
        return pd.DataFrame()

    res = result_list[0]
    ts = res.get("timestamp") or []
    indicators = (res.get("indicators") or {}).get("quote") or []
    if not ts or not indicators:
        return pd.DataFrame()

    q = indicators[0]

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz),
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
        }
    )

    df.insert(0, "symbol", symbol)
    return df.dropna(subset=["datetime"]).copy()


def fetch_yahoo_intraday_one_day(
    session,
    symbol,
    date,
    interval,
    tz="America/New_York",
    user_agent="Mozilla/5.0",
    include_prepost=False,
):
    """
    Lädt Intraday-Daten (ein Tag) für ein Symbol über den Yahoo-Chart-Endpunkt.

    Parameter:
    - session: requests.Session (wird von collect_sp500_intraday_data bereitgestellt)
    - symbol: Yahoo-Ticker
    - date: ISO-Datum (YYYY-MM-DD)
    - interval: Intervall (z. B. '1m')
    - tz: Zielzeitzone für die 'datetime' Spalte
    - user_agent: User-Agent Header
    - include_prepost: ob Pre-/Post-Market inkludiert wird

    Rückgabe:
    - DataFrame (kann leer sein)

    Side effects:
    - Netzwerkzugriff auf Yahoo.
    - r.raise_for_status kann HTTP-Errors als Exception werfen.
    """
    period1, period2 = date_to_unix_periods(date, tz=tz)

    params = {
        "period1": period1,
        "period2": period2,
        "interval": interval,
        "includePrePost": "true" if include_prepost else "false",
        "events": "div|split",
    }
    headers = {"User-Agent": user_agent}

    url = YAHOO_CHART_URL.format(symbol=symbol)
    r = session.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()

    return parse_yahoo_chart_json(symbol, r.json(), tz=tz)


def print_progress(i, total, start_t, rows):
    """
    Gibt eine klassische Progress-Bar in einer Zeile aus.

    Hinweis:
    - Der '\r' Prefix und flush sind Teil des beobachtbaren Verhaltens.
    - Nicht durch print ersetzen, um Zeilenumbrüche/Format nicht zu verändern.
    """
    elapsed = time.time() - start_t
    pct = (i / total) * 100
    msg = (
        "\rProgress: {:6.2f}% | {:4d}/{:4d} | elapsed {:6.1f}s | rows {:8d}"
        .format(pct, i, total, elapsed, rows)
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def collect_sp500_intraday_data(
    date,
    interval="1m",
    universe=None,
    throttle_s=0.1,
    verbose=True,
):
    """
    Sammelt Intraday-Daten für ein S&P 500 Universe (oder ein gegebenes Universe).

    Parameter:
    - date: ISO-Datum (YYYY-MM-DD)
    - interval: Intervall für Yahoo-Chart (z. B. '1m')
    - universe: iterable von Symbolen; wenn None, wird get_sp500_universe() verwendet
    - throttle_s: sleep pro Symbol (auch bei Fehlern) zur Drosselung
    - verbose: wenn True, Statusprints und Progress-Bar

    Rückgabe:
    - DataFrame mit Spalten: symbol, datetime, open, high, low, close, volume
      (oder leerer DataFrame, falls nichts gesammelt wurde)

    Side effects / Besonderheiten:
    - Netzwerkzugriffe auf Yahoo (pro Symbol).
    - Exceptions pro Symbol werden bewusst geschluckt, um fortzufahren.
    - Progress-Bar via sys.stdout.write (eine Zeile) bei verbose=True.
    """
    if universe is None:
        universe = get_sp500_universe(verbose=verbose)

    universe = list(universe)
    total = len(universe)

    if verbose:
        print("Sammle Intraday-Daten")
        print("Datum:", date, "| Interval:", interval)

    all_rows = []
    start_t = time.time()
    total_rows = 0

    with requests.Session() as session:
        for i, symbol in enumerate(universe, start=1):
            try:
                df_sym = fetch_yahoo_intraday_one_day(session, symbol, date, interval)
                if not df_sym.empty:
                    all_rows.append(df_sym)
                    total_rows += len(df_sym)
            except Exception:
                pass
            finally:
                if throttle_s and throttle_s > 0:
                    time.sleep(throttle_s)

            if verbose:
                print_progress(i, total, start_t, total_rows)

    if verbose:
        print()  # Zeilenumbruch nach Progress-Bar

    if not all_rows:
        out = pd.DataFrame()
    else:
        out = pd.concat(all_rows, ignore_index=True)

    if verbose:
        if out.empty:
            print("Keine Daten gesammelt.")
        else:
            print("FERTIG")
            print("Zeilen:", len(out))
            print("Ticker:", out["symbol"].nunique())
            print("Zeitbereich:", out["datetime"].min(), "bis", out["datetime"].max())

    return out


if __name__ == "__main__":
    # Optionaler Script-Entry-Point zur manuellen Ausführung.
    # Hinweis: date default basiert auf lokaler Systemzeit (today()).
    date = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
    interval = sys.argv[2] if len(sys.argv) > 2 else "1m"

    df = collect_sp500_intraday_data(date=date, interval=interval, verbose=True)
    print(df.head(10))