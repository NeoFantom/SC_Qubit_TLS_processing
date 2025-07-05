# Instructions on TLS_DATA

In this specification, `->` means "mapping to", `:` means "type".

The structure of TLS_DATA is as follows:

```text
TLS_DATA: dict {
    qubit: int -> experiment_data: dict = {
        "xs" -> frequencies: list[L] [
            x1: np.ndarray[N1],
            x2: np.ndarray[N2],
            ...
        ],
        "ys" -> strengths: list[L] [
            y1: np.ndarray[N1],
            y2: np.ndarray[N2],
            ...
        ],
        "times" -> times_stamps: list[L] [
            time1: str,
            time2: str,
            ...
        ]
    },
    qubit: int -> experiment_data: dict = {
        ...
    },
    ...
}
```

Each time in `TLS_DATA[q]["times"]` is a Python string that can be converted to a date using:
```python
datetime.strptime(t, '%a %b %d %H:%M:%S %Y').date()
```
