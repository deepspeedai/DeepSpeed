#pragma once
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#ifdef _WIN32

class Stopwatch {
private:
    double m_total_time;
    LARGE_INTEGER m_start_time;

public:
    Stopwatch() { m_total_time = 0.0; }

    ~Stopwatch() {}

    void Reset() { m_total_time = 0.0; }

    void Start() { QueryPerformanceCounter(&m_start_time); }

    void Restart()
    {
        m_total_time = 0.0;
        QueryPerformanceCounter(&m_start_time);
    }

    void Stop()
    {
        LARGE_INTEGER frequency;
        LARGE_INTEGER stop_time;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&stop_time);
        m_total_time +=
            ((double)(stop_time.QuadPart - m_start_time.QuadPart) / (double)frequency.QuadPart);
    }

    double GetTimeInSeconds() { return m_total_time; }
};

#else

class Stopwatch {
private:
    double m_total_time;
    struct timespec m_start_time;
    bool m_is_started;

public:
    Stopwatch()
    {
        m_total_time = 0.0;
        m_is_started = false;
    }

    ~Stopwatch() {}

    void Reset() { m_total_time = 0.0; }

    void Start()
    {
        clock_gettime(CLOCK_MONOTONIC, &m_start_time);
        m_is_started = true;
    }

    void Restart()
    {
        m_total_time = 0.0;
        clock_gettime(CLOCK_MONOTONIC, &m_start_time);
        m_is_started = true;
    }

    void Stop()
    {
        if (m_is_started) {
            m_is_started = false;

            struct timespec end_time;
            clock_gettime(CLOCK_MONOTONIC, &end_time);

            m_total_time += (double)(end_time.tv_sec - m_start_time.tv_sec) +
                            (double)(end_time.tv_nsec - m_start_time.tv_nsec) / 1e9;
        }
    }

    double GetTimeInSeconds()
    {
        if (m_is_started) {
            Stop();
            Start();
        }
        return m_total_time;
    }
};

#endif
