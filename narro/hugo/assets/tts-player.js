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
  var rafId = null;

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
   * Normalize a word for matching: lowercase, strip punctuation.
   */
  function normalizeWord(w) {
    return w.toLowerCase().replace(/[^a-z0-9']/g, "");
  }

  /**
   * Walk text nodes in the article, wrap each word in a <span>.
   * Match words to alignment entries by normalized text comparison.
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

        // Match DOM word to alignment entry by normalized text
        if (alignIdx < alignment.length) {
          var domNorm = normalizeWord(part);
          var alignNorm = normalizeWord(alignment[alignIdx].word);
          if (domNorm === alignNorm || domNorm.indexOf(alignNorm) >= 0 || alignNorm.indexOf(domNorm) >= 0) {
            span.dataset.idx = alignIdx;
            wordSpans[alignIdx] = span;
            alignIdx++;
          }
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
        // Scroll active word into view if off-screen
        scrollToWord(wordSpans[idx]);
      }
      activeIdx = idx;
    }
  }

  /**
   * Gently scroll to keep the active word visible.
   */
  function scrollToWord(span) {
    var rect = span.getBoundingClientRect();
    var viewH = window.innerHeight;
    // Only scroll if the word is outside the middle 60% of the viewport
    if (rect.top < viewH * 0.2 || rect.bottom > viewH * 0.8) {
      span.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  /**
   * Animation loop for smooth highlighting (~60fps while playing).
   */
  function tick() {
    updateTime();
    highlightWord();
    if (!audio.paused) {
      rafId = requestAnimationFrame(tick);
    }
  }

  function startLoop() {
    if (rafId === null) {
      rafId = requestAnimationFrame(tick);
    }
  }

  function stopLoop() {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    updateTime();
    highlightWord();
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

  audio.addEventListener("play", startLoop);
  audio.addEventListener("pause", stopLoop);
  audio.addEventListener("seeked", function () {
    updateTime();
    highlightWord();
  });

  audio.addEventListener("ended", function () {
    stopLoop();
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
