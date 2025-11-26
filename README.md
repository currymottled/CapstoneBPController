Control systems are ubiqitous and 'digital twins' are becoming common. The medical versions of these are defective for a few reasons. Human bodies (and animals in general) do
not come with datasheets and are not made to be probed by test kids, reverse engineered, or taken apart for maintenance; most medicine is risky patchwork. This project starts off
with the notion that the missing element in medical automation is physiologically accurate models with parameters that can be reliabily estimated and controlled.

Anaesthesiologists watch over the homeostasis of several different body systems at once during operations and intensive care stays, often including the respiratory and cardiovascular
systems of patients on top of their central nervous systems (anaesthesia). Typically they have to delegate, and in any case manual oversight is prone to error. 

It turns out that there are a variety of 'Windkessel' circuit-analogue models for the cardiovascular system, which may be accurate enough for regulation. There are a couple problems, 
however. First, these circuits are not like familiar manufactured circuits in that their 'component' values are unknown and prone to drift. The other problem is that it is hard to
locate or 'measure' these 'components' because what they represent is inherently distributed. A minor issue is that the analogy between bloodflow and electricity is a bit dubious.

In point of fact, what actually matters for hospitals is that the models work. There are no circuits involved, really, only difference equations and digital signal processing to be 
handled by computers. What is needed for that is trials to show the physiological accuracy of different models and the effect of cardiac drugs on model parameters. Unfortunately, that 
is out of the scope of a school project, but it is possible to show that standard control approaches (i.e., LQR) would work fine on realistic inputs provided known relationships.

There are a few puzzle pieces to this control system:
## Major BP Regulation Model Components

| Project Component      | Approach                                      |
|------------------------|-----------------------------------------------|
| Input                  | Blood flow waveform                           |
| Output                 | Blood pressure waveform                       |
| Live State Estimation  | Signal processing & analysis (blood pressure & flow) |
| Actuators              | Cardiac drugs                                 |

The system as a whole has more parts (sensors, database, data relay etc.), but these issues are routine. This project will include a database for storing results, sensors to use as 
inputs alongside simulated inputs (heart rate, electrocardiogram), and a dashboard. These are largely side-tasks, however.

For now the structure of the files looks like this:
## Project Structure

| File               | Purpose                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `config.py`        | Parameters/constants to be tuned, imported in all other files           |
| `pump.py`          | Generates input blood flow                                              |
| `windkessel.py`    | Calculates BP waveform based on blood flow and a three‑element Windkessel (Za, R, C) |
| `signal_process.py`| Estimates (Za, R, C) as if unknown with signal processing from windkessel.py |
| `control.py`       | Controller – takes in processed BP waveform from signal_process.py and controls drug infusion |
| `pk.py`            | Pharmacokinetic model – calculates drug concentration in plasma over time with a two‑compartment model based on infusion |
| `pd.py`            | Pharmacodynamic model – calculates the effect of drug concentration levels on Windkessel parameters |


If this first cut succeeds the following would be nice:
  - Higher element Windkessel (say 5 elements)
  - Simulated effect of drugs on input blood flow (unintended feedback)
  - Simulated effects of a large profile of cardiac drugs with mixed effects on both the heart and vascular system
  - Blood flow (CO) control rather than only vasocative control of SVR
  - Predifined scenario testing
  - Reliability and robustness testing

It will likely be wise to test a variety of plausible relationships between variables since actual relationships seem to be as of yet mysterious to the medical and academic community.
Either way, the main point is to show that standard approaches are not acutely sensitive to the exact nature of the cardiovascular system; i.e. once the variables are pinned down and
good enough, control will work. Thank you for reading and wish us luck.


  

  
