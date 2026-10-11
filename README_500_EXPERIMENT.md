# 500-vehicle VANET cyberattack experiment

Workspace: `/home/bindu/VANET_500_Experiment`  
Protected (never modified by this experiment): `/home/bindu/VANET_Project`, `/home/bindu/VANET_Project_BACKUP`

Simulation: NS-3.48, `scratch/vanet_500.cc`  
Dataset: FlowMonitor flow-level records (not one row per vehicle and not one row per packet)

## What was actually implemented

500 vehicle nodes plus 1 RSU on a 25-column grid. IEEE 802.11p ad hoc Wi-Fi, `/16` IPv4 subnet, no AODV. Radio range is 400 m (`RangePropagationLossModel`). Legitimate UDP senders are chosen **inside 300 m** of the RSU (direct) or the relay (hop) so packets can actually be received.

| AttackType | Scenario | Behaviour in this architecture |
|---|---|---|
| 0 | Normal | 40 vehicles send UDP to the RSU. 20 vehicles send UDP to vehicle **497**, which **relays** to the RSU. |
| 1 | DoS | Same as Normal, plus vehicle **499** sends 1024-byte UDP to the RSU every 1 ms (high-rate). Detection uses FlowMonitor `TxPackets > 500`, not a console-only flag. |
| 2 | Sybil | **Application-level only.** Vehicle **498** sends 5 extra UDP flows to the RSU on ports 6001–6005 (claimed identities). The NS-3 node ID does **not** change. Limitation: this is not a protocol-level (IEEE 1609 / pseudonym) Sybil attack. |
| 3 | Blackhole | **Application-level drop on a deliberate forwarding path.** Vehicle **497** receives hop-client UDP and **drops** it instead of forwarding. Verified with relay counters (`rx>0`, `fwd=0`, `dropped>0`) and lower RSU sink bytes than Normal. This is **not** an AODV/IP routing blackhole. Direct-to-RSU flows are not dropped. |

## Why the original XMLs had almost no flows

The preserved files `results/vanet500-flowmon-0.xml` and `vanet500-flowmon-1.xml` contain 1 and 2 FlowMonitor flows. The old extraction used `.//Flow`, which also matches classifier rows, and those runs used almost no senders. Those files were **not overwritten**.

## Build

```bash
cd /home/bindu/VANET_500_Experiment/ns-3.48
./ns3 build vanet_500
```

## Run one scenario

```bash
cd /home/bindu/VANET_500_Experiment/ns-3.48
./run_vanet_500.sh 0 500 1 1 true     # Normal
./run_vanet_500.sh 1 500 1 1 true     # DoS
./run_vanet_500.sh 2 500 1 1 true     # Sybil
./run_vanet_500.sh 3 500 1 1 true     # Blackhole
```

Arguments: `ATTACK_TYPE NUM_VEHICLES RUN_ID RNG_SEED ENABLE_ANIM [ENABLE_PKT_META]`

Outputs (unique names; legacy XMLs kept):

- `results/vanet500_r{RUN}_s{SEED}_a{ATTACK}_flowmon.xml`
- `results/vanet500_r{RUN}_s{SEED}_a{ATTACK}_animation.xml`
- `results/vanet500_r{RUN}_s{SEED}_a{ATTACK}_meta.csv`

## Extra genuine runs

```bash
cd /home/bindu/VANET_500_Experiment/ns-3.48
./run_vanet_500_batch.sh 10 true
```

That script runs seeds 11 and 22 for all four attacks with new RunIDs starting at 10.

## Dataset extraction

```bash
cd /home/bindu/VANET_500_Experiment/ns-3.48
python3 extract_dataset_500.py
```

Writes:

- `/home/bindu/VANET_500_Experiment/dataset/vanet_500_dataset.csv`
- `/home/bindu/VANET_500_Experiment/dataset/dataset_summary.csv`
- `dataset/extraction_per_xml.csv`
- `dataset/run_seed_map.csv`

Labels are taken from filenames / `_meta.csv`, never inferred from FlowID.  
`PacketLossRate = (TxPackets - RxPackets) / TxPackets * 100` when TxPackets > 0.  
Delay, jitter, and throughput are left **empty** when there are no received packets (not replaced with 0).

## Dataset validation

```bash
cd /home/bindu/VANET_500_Experiment/ns-3.48
python3 validate_dataset_500.py
```

Report: `results/dataset_validation_500.txt`  
No Random Forest, SVM, or Streamlit is included in this step.

## Verified execution (this workspace)

| RunID | Seed | Scenarios |
|------|------|-----------|
| 0 (legacy XML) | unknown | Normal, DoS only; Tx>0 but Rx=0 |
| 1 | 1 | Normal, DoS, Sybil, Blackhole |
| 10–13 | 11 | Normal, DoS, Sybil, Blackhole |
| 14–17 | 22 | Normal, DoS, Sybil, Blackhole |

Example measured evidence (RunID=1, seed=1):

- Normal: 60 flows, 43 with Rx>0; relay rx=354 fwd=354 dropped=0; RSU sink 631296 bytes
- DoS: 61 flows; attacker flow Tx=6000 Rx=1856 (threshold 500)
- Sybil: 65 flows (60 legitimate + 5 claimed-identity ports)
- Blackhole: relay rx=354 fwd=0 dropped=354; RSU sink 436224 bytes (hop traffic not delivered)

## Genuine records in the combined CSV

Total **741** flow-level records, 0 duplicates.

| Class | AttackType | Rows |
|---|---|---|
| Normal | 0 | 181 |
| DoS | 1 | 185 |
| Sybil | 2 | 195 |
| Blackhole | 3 | 180 |

Measurement level: **flow**. 500 vehicles do not produce 500 independent rows.

## Known limitations

1. **Sybil** is application-level extra UDP identities (destination ports). Node IDs are unchanged.
2. **Blackhole** is an application relay drop, not a routing-protocol blackhole. Packets that go direct to the RSU are not eaten by vehicle 497.
3. Some flows have Tx>0 and Rx=0 because of 802.11p contention and range; delay/jitter/throughput are missing for those rows.
4. NetAnim may still print `Max Packets per trace file exceeded`. Animation files can be tens of MB. Use `--enablePktMeta=false` (default) and optionally `--enableAnim=false` for faster extra runs.
5. Legacy `vanet500-flowmon-0.xml` / `-1.xml` are included with RunID=0 and missing RandomSeed.
6. FlowMonitor `lostPackets` is stored as `LostPackets`; it can under-count relative to Tx−Rx. Use `PacketLossRate` for delivery-based loss.

## Continue from here

```bash
cd /home/bindu/VANET_500_Experiment/ns-3.48
python3 extract_dataset_500.py
python3 validate_dataset_500.py
# optional extra seed:
./run_vanet_500.sh 0 500 20 33 true
# ML (Random Forest / SVM / Streamlit) is intentionally not implemented yet
```
