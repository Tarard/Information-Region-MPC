"""Reuse the saved native signal executor without changing its rules."""

import math

def integer_green(values, signal):
    budget = int(signal['available_green_seconds'])
    nominal = list(map(float, signal['nominal_green_seconds']))
    if len(values) != len(signal['green_native_indices']) or len(values) != len(nominal):
        raise ValueError('MPC green vector does not match native main phases')
    if not all(math.isfinite(x) for x in values) or abs(sum(values) - budget) > 1e-4:
        raise ValueError('MPC green vector does not satisfy the cycle budget')
    if any(x < (6 if old < 10 else 10) - 1e-4 or x > 60 + 1e-4 for x, old in zip(values, nominal)):
        raise ValueError('MPC green vector violates the declared bounds')
    whole = [math.floor(x + 1e-8) for x in values]
    remaining = budget - sum(whole)
    if not 0 <= remaining <= len(values):
        raise ValueError('Cannot round MPC greens within the cycle')
    order = sorted(range(len(values)), key=lambda i: (-(values[i] - whole[i]), i))
    for i in order[:remaining]:
        whole[i] += 1
    if sum(whole) != budget or any(x < (6 if old < 10 else 10) or x > 60 for x, old in zip(whole, nominal)):
        raise ValueError('Rounded greens violate the signal plan')
    return whole


def schedule(signal, action, start):
    greens = integer_green(action, signal)
    main = dict(zip(signal['green_native_indices'], greens))
    phases = []
    now = start
    for index, phase in enumerate(signal['phases']):
        length = main.get(index, int(phase['duration_seconds']))
        phases.append({'phase': index, 'start': now, 'end': now + length, 'state': phase['state'],
                       'kind': 'optimized_green' if index in main else 'native_transition'})
        now += length
    if now - start != int(signal['cycle_seconds']):
        raise ValueError('Executed cycle does not retain native total duration')
    return {'start': start, 'end': now, 'requested_green_seconds': action,
            'executed_green_seconds': greens, 'phases': phases}


def expected_state(plan, now):
    return next(p['state'] for p in plan['phases'] if p['start'] <= now < p['end'])


def legal_green(values, signal):
    nominal = list(map(float, signal['nominal_green_seconds']))
    lower = [6.0 if x < 10 else 10.0 for x in nominal]
    upper = [60.0] * len(values)
    budget = float(signal['available_green_seconds'])
    if abs(sum(values) - budget) > 0.01 or any(x < lo - 0.01 or x > hi + 0.01
                                                for x, lo, hi in zip(values, lower, upper)):
        raise ValueError('Solver action is too far from the legal green set')
    corrected = [min(max(x, lo), hi) for x, lo, hi in zip(values, lower, upper)]
    delta = budget - sum(corrected)
    for _ in corrected:
        if abs(delta) < 1e-8:
            break
        slack = [(hi - x) if delta > 0 else (x - lo)
                 for x, lo, hi in zip(corrected, lower, upper)]
        i = max(range(len(slack)), key=slack.__getitem__)
        if slack[i] < 1e-8:
            raise ValueError('No room to repair numerical green error')
        change = math.copysign(min(abs(delta), slack[i]), delta)
        corrected[i] += change
        delta -= change
    if abs(sum(corrected) - budget) > 1e-8:
        raise ValueError('Legal green repair did not meet the cycle budget')
    return corrected, max(abs(x - y) for x, y in zip(values, corrected))


