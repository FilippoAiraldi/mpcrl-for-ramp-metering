NO CONTROL
    No OPTI/NLP. Local and colab are the same obviously.

RAMP ONLY
    No substantial difference in OPTI and NLP, OPTI is a bit better to look at.
    However, with equal settings, OPTI runs in 2:41 instead of 1:03 of NLP.
    No substantial difference between local and colab (here must be run with 
    20s max cpu time, otherwise crashes).

    With which settings does the NLP achieve same precision as OPTI but in less
    time?
    - Never, not even with 30s of max cpu time. There's always a (probably 
      incorrect) spike of the ramp control.

COORDINATED CONTROL (RAMP + VMS)
    On local laptop, NLP does not deliver the right answer (and runs very 
    quickly 3:28). With the same settings, OPTI runs good locally and produces 
    acceptable results (colab's are better but run longer).
    On colab, NLP delivers very good in 13:33. OPTI needs again 20s max wall 
    time not to crash, and yields bit worse results in 30:26.

    With which settings does the coordinated control run on local with NLP?
    - At least 30s max cpu time


no_ctrl and ramp_only are equal both on colab,