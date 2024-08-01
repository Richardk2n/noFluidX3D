void __attribute__((always_inline)) atomicAdd(volatile global double* targetAddress, const double valueToAdd) {
	union {
		ulong  asLong;
		double asDouble;
	} currentValue, lastValue, targetValue;
	currentValue.asDouble = *targetAddress;
	do {
		targetValue.asDouble = (lastValue.asDouble=currentValue.asDouble)+valueToAdd;  //TODO convert to explicit
		currentValue.asLong = atom_cmpxchg((volatile global ulong*)targetAddress, lastValue.asLong, targetValue.asLong);
	} while(currentValue.asLong!=lastValue.asLong);
}
