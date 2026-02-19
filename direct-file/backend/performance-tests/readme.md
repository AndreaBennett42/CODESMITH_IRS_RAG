**Install Jmeter**
- ```brew install jmeter```

- ```jmeter -v```

**Start JMeter GUI**
- open terminal
- run command ```jmeter```

**Running Jmeter Test via GUI**
- start jmeter GUI
- open the desired ```.jmx``` file
- press the green play button

**Open Log Viewer in GUI**
- From the menu bar, enable the LogViewer via: ```Options -> LogViewer```

**Troubleshooting**
- If logs stop appearing in the console, restart JMeter.

**Props vs Vars**
- ```props``` are global key-value pairs that can be shared between threads.
- ```vars``` are thread-level key-value pairs that are cannot be shared between other threads.

**User Defined Variables vs User Parameters**
- User Defined Variables are set globally, and can be accessed by all threads.
- User Parameters are recomputed for every thread or optionally for very iteration

**Open Issues**
- for some reason, when I close JMeter and reopen it, and then run a test, the variable fails to evaluate