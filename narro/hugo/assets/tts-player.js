/**
 * TTS Player — word-level highlighting synchronized with audio playback.
 * Installed by: narro hugo install
 */
(function () {
  "use strict";

  var player = document.querySelector(".tts-player");
  if (!player) return;

  var audio = player.querySelector("audio");
  var btn = player.querySelector(".tts-play");
  var timeDisplay = player.querySelector(".tts-time");
  var alignUrl = player.dataset.align;

  var alignment = [];
  var wordSpans = [];
  var activeIdx = -1;

  // --- Initialization ---

  function init() {
    if (alignUrl) {
      fetch(alignUrl)
        .then(function (r) { return r.json(); })
        .then(function (data) {
          alignment = data;
          wrapWords();
        })
        .catch(function () {}); // Degrade gracefully: no highlighting
    }
  }

  /**
   * Walk text nodes in the article, wrap each word in a <span>.
   * Match words sequentially to alignment entries.
   */
  function wrapWords() {
    var article = document.querySelector("article") || document.querySelector(".content");
    if (!article || alignment.length === 0) return;

    var alignIdx = 0;
    var walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT);
    var textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);

    for (var i = 0; i < textNodes.length; i++) {
      var node = textNodes[i];
      var text = node.textContent;
      if (!text.trim()) continue;

      var frag = document.createDocumentFragment();
      var parts = text.split(/(\s+)/);

      for (var j = 0; j < parts.length; j++) {
        var part = parts[j];
        if (/^\s+$/.test(part)) {
          frag.appendChild(document.createTextNode(part));
          continue;
        }
        if (!part) continue;

        var span = document.createElement("span");
        span.className = "tts-word";

        if (alignIdx < alignment.length) {
          span.dataset.idx = alignIdx;
          wordSpans[alignIdx] = span;
          alignIdx++;
        }

        span.textContent = part;
        span.addEventListener("click", onWordClick);
        frag.appendChild(span);
      }

      node.parentNode.replaceChild(frag, node);
    }
  }

  // --- Playback ---

  function formatTime(s) {
    var m = Math.floor(s / 60);
    var sec = Math.floor(s % 60);
    return m + ":" + (sec < 10 ? "0" : "") + sec;
  }

  function updateTime() {
    if (!audio.duration) return;
    timeDisplay.textContent =
      formatTime(audio.currentTime) + " / " + formatTime(audio.duration);
  }

  /**
   * Binary search alignment array for the word active at currentTime.
   */
  function highlightWord() {
    if (alignment.length === 0) return;

    var t = audio.currentTime;
    var lo = 0,
      hi = alignment.length - 1,
      mid,
      idx = -1;

    while (lo <= hi) {
      mid = (lo + hi) >> 1;
      if (alignment[mid].start <= t) {
        idx = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }

    // Check if time is within the found word's range
    if (idx >= 0 && t > alignment[idx].end) idx = -1;

    if (idx !== activeIdx) {
      if (activeIdx >= 0 && wordSpans[activeIdx]) {
        wordSpans[activeIdx].classList.remove("tts-active");
      }
      if (idx >= 0 && wordSpans[idx]) {
        wordSpans[idx].classList.add("tts-active");
      }
      activeIdx = idx;
    }
  }

  function onWordClick(e) {
    var idx = parseInt(e.target.dataset.idx, 10);
    if (!isNaN(idx) && idx < alignment.length) {
      audio.currentTime = alignment[idx].start;
      if (audio.paused) audio.play();
    }
  }

  // --- Controls ---

  btn.addEventListener("click", function () {
    if (audio.paused) {
      audio.play();
      btn.textContent = "\u275A\u275A Pause";
    } else {
      audio.pause();
      btn.textContent = "\u25B6 Listen";
    }
  });

  audio.addEventListener("timeupdate", function () {
    updateTime();
    highlightWord();
  });

  audio.addEventListener("ended", function () {
    btn.textContent = "\u25B6 Listen";
    if (activeIdx >= 0 && wordSpans[activeIdx]) {
      wordSpans[activeIdx].classList.remove("tts-active");
    }
    activeIdx = -1;
  });

  document.addEventListener("keydown", function (e) {
    if (e.code === "Space" && e.target === document.body) {
      e.preventDefault();
      btn.click();
    }
  });

  init();
})();
