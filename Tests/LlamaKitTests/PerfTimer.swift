//
//  PerfTimer.swift
//  AutoReminder
//
//  Created by Prashanth Sadasivan on 8/19/24.
//

import Foundation

class PerfTimer {
    var startTime = Date()
    var accumulation = TimeInterval()
    
    func lap(desc: String) -> TimeInterval {
        let endTime = Date()
        let timeElapsed = endTime.timeIntervalSince(startTime)
        print("Lap: \(desc): \(timeElapsed) seconds")
        accumulation += timeElapsed
        startTime = endTime
        return timeElapsed
    }
    
    func total() -> TimeInterval{
        print("Total Lap time: \(accumulation)")
        return accumulation;
    }
}
