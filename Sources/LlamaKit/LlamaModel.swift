//
//  LLamaModel.swift
//  LLLists
//
//  Created by Prashanth Sadasivan on 6/17/24.
//

import Foundation
import SwiftData


@Model
class LlamaModel: Identifiable{
    var id = UUID()
    var name: String
    var nameOfFile: String
    var url: String
    var systemPrompt: String
    var userPromptPostfix: String
    var selected: Bool
    var pathToFile: String?
    init(id: UUID = UUID(), name: String, nameOfFile: String, url: String, system: String, user: String, selected: Bool, pathToFile: String? = nil) {
        self.id = id
        self.name = name
        self.nameOfFile = nameOfFile
        self.url = url
        self.systemPrompt = system
        self.userPromptPostfix = user
        self.selected = selected
        self.pathToFile = pathToFile
    }
   
}
