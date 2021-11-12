/*
* Copyright 2021 Google LLC
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     https://www.apache.org/licenses/LICENSE-2.0
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* This file contains a JSON formatted dictionary of available words:
* - Key corresponds to the a unique reference for the
*     single word spoken in the videos; if it is a duplicate word, annotate it
*     with a parenthesized number, e.g. pan (1)
* - "Word" is the word spoken in the video.
* - "TrainVideos" is a list of filenames for the videos that should be used
*   in the "training" and "practice" modes.
* - "TestVideos" is a list of filenames for the videos that should be used
*   in the "test" mode.
* - "Grouping" is a list of words that should be included in a multiple choice
*   question for this word.  This always includes the word itself.
*
* To add more videos, add an entry to the list, and include
* the videos with the same filenames in the appropriate media
* folder (defined as basePath in the main file).
*
* To include a word multiple times with separate videos, annotate the key with
* a parenthetical number, e.g:
*    "pan (1)":{
*      "word":"pan",
*      "trainVideos":["pan4.mp4", "pan5.mp4"],
*      "testVideos":["pan6.mp4"],
*      "grouping":["pan (1)", "man"]
*    }
*/

wordsAndVideos = {
  "pan":{
    "word":"pan",
    "trainVideos":["pan1.mp4", "pan2.mp4"],
    "testVideos":["pan3.mp4"],
    "grouping":["pan", "ban"]
  },

  "ban":{
    "word":"ban",
    "trainVideos":["ban1.mp4", "ban2.mp4"],
    "testVideos":["ban3.mp4"],
    "grouping":["ban", "pan"]
  },

  "pea":{
    "word":"pea",
    "trainVideos":["pea1.mp4", "pea2.mp4"],
    "testVideos":["pea3.mp4"],
    "grouping":["pea", "bee"]
  },

  "bee":{
    "word":"bee",
    "trainVideos":["bee1.mp4", "bee2.mp4"],
    "testVideos":["bee3.mp4"],
    "grouping":["bee", "pea"]
  },

  "fan":{
    "word":"fan",
    "trainVideos":["fan1.mp4", "fan2.mp4"],
    "testVideos":["fan3.mp4"],
    "grouping":["fan", "van"]
  },

  "van":{
    "word":"van",
    "trainVideos":["van1.mp4", "van2.mp4"],
    "testVideos":["van3.mp4"],
    "grouping":["van", "fan"]
  },

  "fee":{
    "word":"fee",
    "trainVideos":["fee1.mp4", "fee2.mp4"],
    "testVideos":["fee3.mp4"],
    "grouping":["fee", "vee"]
  },

  "vee":{
    "word":"vee",
    "trainVideos":["vee1.mp4", "vee2.mp4"],
    "testVideos":["vee3.mp4"],
    "grouping":["vee", "fee"]
  },

  "thigh":{
    "word":"thigh",
    "trainVideos":["thigh1.mp4", "thigh2.mp4"],
    "testVideos":["thigh3.mp4"],
    "grouping":["thigh", "thy"]
  },

  "thy":{
    "word":"thy",
    "trainVideos":["thy1.mp4", "thy2.mp4"],
    "testVideos":["thy3.mp4"],
    "grouping":["thy", "thigh"]
  },

  "sip":{
    "word":"sip",
    "trainVideos":["sip1.mp4", "sip2.mp4"],
    "testVideos":["sip3.mp4"],
    "grouping":["sip", "zip"]
  },

  "zip":{
    "word":"zip",
    "trainVideos":["zip1.mp4", "zip2.mp4"],
    "testVideos":["zip3.mp4"],
    "grouping":["zip", "sip"]
  },

  "sap":{
    "word":"sap",
    "trainVideos":["sap1.mp4", "sap2.mp4"],
    "testVideos":["sap3.mp4"],
    "grouping":["sap", "zap"]
  },

  "zap":{
    "word":"zap",
    "trainVideos":["zap1.mp4", "zap2.mp4"],
    "testVideos":["zap3.mp4"],
    "grouping":["zap", "sap"]
  },

  "dell":{
    "word":"dell",
    "trainVideos":["dell1.mp4", "dell2.mp4"],
    "testVideos":["dell3.mp4"],
    "grouping":["dell", "tell"]
  },

  "tell":{
    "word":"tell",
    "trainVideos":["tell1.mp4", "tell2.mp4"],
    "testVideos":["tell3.mp4"],
    "grouping":["tell", "dell"]
  },

  "shock":{
    "word":"shock",
    "trainVideos":["shock1.mp4", "shock2.mp4"],
    "testVideos":["shock3.mp4"],
    "grouping":["shock", "jacques"]
  },

  "jacques":{
    "word":"jacques",
    "trainVideos":["jacques1.mp4", "jacques2.mp4"],
    "testVideos":["jacques3.mp4"],
    "grouping":["jacques", "shock"]
  },

  "cold":{
    "word":"cold",
    "trainVideos":["cold1.mp4", "cold2.mp4"],
    "testVideos":["cold3.mp4"],
    "grouping":["cold", "gold"]
  },

  "gold":{
    "word":"gold",
    "trainVideos":["gold1.mp4", "gold2.mp4"],
    "testVideos":["gold3.mp4"],
    "grouping":["gold", "cold"]
  },

  "cap":{
    "word":"cap",
    "trainVideos":["cap1.mp4", "cap2.mp4"],
    "testVideos":["cap3.mp4"],
    "grouping":["cap", "gap"]
  },

  "gap":{
    "word":"gap",
    "trainVideos":["gap1.mp4", "gap2.mp4"],
    "testVideos":["gap3.mp4"],
    "grouping":["gap", "cap"]
  },

  "dee":{
    "word":"dee",
    "trainVideos":["dee1.mp4", "dee2.mp4"],
    "testVideos":["dee3.mp4"],
    "grouping":["dee", "key"]
  },

  "key":{
    "word":"key",
    "trainVideos":["key1.mp4", "key2.mp4"],
    "testVideos":["key3.mp4"],
    "grouping":["key", "dee"]
  },

  "dog":{
    "word":"dog",
    "trainVideos":["dog1.mp4", "dog2.mp4"],
    "testVideos":["dog3.mp4"],
    "grouping":["dog", "cog"]
  },

  "cog":{
    "word":"cog",
    "trainVideos":["cog1.mp4", "cog2.mp4"],
    "testVideos":["cog3.mp4"],
    "grouping":["cog", "dog"]
  },

  "too":{
    "word":"too",
    "trainVideos":["too1.mp4", "too2.mp4"],
    "testVideos":["too3.mp4"],
    "grouping":["too", "goo"]
  },

  "goo":{
    "word":"goo",
    "trainVideos":["goo1.mp4", "goo2.mp4"],
    "testVideos":["goo3.mp4"],
    "grouping":["goo", "too"]
  },

  "tie":{
    "word":"tie",
    "trainVideos":["tie1.mp4", "tie2.mp4"],
    "testVideos":["tie3.mp4"],
    "grouping":["tie", "guy"]
  },

  "guy":{
    "word":"guy",
    "trainVideos":["guy1.mp4", "guy2.mp4"],
    "testVideos":["guy3.mp4"],
    "grouping":["guy", "tie"]
  },

  "tell (1)":{
    "word":"tell",
    "trainVideos":["tell4.mp4", "tell5.mp4"],
    "testVideos":["tell6.mp4"],
    "grouping":["tell (1)", "yell"]
  },

  "yell":{
    "word":"yell",
    "trainVideos":["yell1.mp4", "yell2.mp4"],
    "testVideos":["yell3.mp4"],
    "grouping":["yell", "tell (1)"]
  },

  "too (1)":{
    "word":"too",
    "trainVideos":["too4.mp4", "too5.mp4"],
    "testVideos":["too6.mp4"],
    "grouping":["too (1)", "you"]
  },

  "you":{
    "word":"you",
    "trainVideos":["you1.mp4", "you2.mp4"],
    "testVideos":["you3.mp4"],
    "grouping":["you", "too (1)"]
  },

  "ken":{
    "word":"ken",
    "trainVideos":["ken1.mp4", "ken2.mp4"],
    "testVideos":["ken3.mp4"],
    "grouping":["ken", "yen"]
  },

  "yen":{
    "word":"yen",
    "trainVideos":["yen1.mp4", "yen2.mp4"],
    "testVideos":["yen3.mp4"],
    "grouping":["yen", "ken"]
  },

  "yolk":{
    "word":"yolk",
    "trainVideos":["yolk1.mp4", "yolk2.mp4"],
    "testVideos":["yolk3.mp4"],
    "grouping":["yolk", "coke"]
  },

  "coke":{
    "word":"coke",
    "trainVideos":["coke1.mp4", "coke2.mp4"],
    "testVideos":["coke3.mp4"],
    "grouping":["coke", "yolk"]
  },

  "man":{
    "word":"man",
    "trainVideos":["man1.mp4", "man2.mp4"],
    "testVideos":["man3.mp4"],
    "grouping":["man", "pan (1)"]
  },

  "pan (1)":{
    "word":"pan",
    "trainVideos":["pan4.mp4", "pan5.mp4"],
    "testVideos":["pan6.mp4"],
    "grouping":["pan (1)", "man"]
  },

  "me":{
    "word":"me",
    "trainVideos":["me1.mp4", "me2.mp4"],
    "testVideos":["me3.mp4"],
    "grouping":["me", "pea (1)"]
  },

  "pea (1)":{
    "word":"pea",
    "trainVideos":["pea4.mp4", "pea5.mp4"],
    "testVideos":["pea6.mp4"],
    "grouping":["pea (1)", "me"]
  },

  "cat":{
    "word":"cat",
    "trainVideos":["cat1.mp4", "cat2.mp4"],
    "testVideos":["cat3.mp4"],
    "grouping":["cat", "gnat"]
  },

  "gnat":{
    "word":"gnat",
    "trainVideos":["gnat1.mp4", "gnat2.mp4"],
    "testVideos":["gnat3.mp4"],
    "grouping":["gnat", "cat"]
  },

  "nit":{
    "word":"nit",
    "trainVideos":["nit1.mp4", "nit2.mp4"],
    "testVideos":["nit3.mp4"],
    "grouping":["nit", "kit"]
  },

  "kit":{
    "word":"kit",
    "trainVideos":["kit1.mp4", "kit2.mp4"],
    "testVideos":["kit3.mp4"],
    "grouping":["kit", "nit"]
  },

  "nap":{
    "word":"nap",
    "trainVideos":["nap1.mp4", "nap2.mp4"],
    "testVideos":["nap3.mp4"],
    "grouping":["nap", "tap"]
  },

  "tap":{
    "word":"tap",
    "trainVideos":["tap1.mp4", "tap2.mp4"],
    "testVideos":["tap3.mp4"],
    "grouping":["tap", "nap"]
  },

  "toe":{
    "word":"toe",
    "trainVideos":["toe1.mp4", "toe2.mp4"],
    "testVideos":["toe3.mp4"],
    "grouping":["toe", "no"]
  },

  "no":{
    "word":"no",
    "trainVideos":["no1.mp4", "no2.mp4"],
    "testVideos":["no3.mp4"],
    "grouping":["no", "toe"]
  }
};
