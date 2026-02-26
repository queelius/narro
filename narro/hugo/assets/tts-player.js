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
  var lastScrollTime = 0;

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
   * Elements whose text should NOT be wrapped for highlighting.
   * These are stripped by extract_prose() and absent from alignment data.
   */
  var SKIP_TAGS = {
    H1:1, H2:1, H3:1, H4:1, H5:1, H6:1,
    PRE:1, CODE:1, SCRIPT:1, STYLE:1, NAV:1,
    FIGCAPTION:1, BUTTON:1, SUMMARY:1
  };

  /**
   * Check if a node is inside an element that should be skipped.
   */
  function isInsideSkipped(node) {
    var el = node.parentNode;
    while (el && el !== document.body) {
      if (SKIP_TAGS[el.tagName]) return true;
      el = el.parentNode;
    }
    return false;
  }

  /**
   * Walk text nodes in the article content, wrap each word in a <span>.
   * Only wraps text inside prose elements (skips headings, code, nav).
   * Words are matched sequentially to alignment entries.
   */
  function wrapWords() {
    var article = document.querySelector(".content") || document.querySelector("article");
    if (!article || alignment.length === 0) return;

    var alignIdx = 0;
    var walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT);
    var textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);

    for (var i = 0; i < textNodes.length; i++) {
      var node = textNodes[i];
      var text = node.textContent;
      if (!text.trim()) continue;
      if (isInsideSkipped(node)) continue;

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
        // Scroll active word into view if off-screen
        scrollToWord(wordSpans[idx]);
      }
      activeIdx = idx;
    }
  }

  /**
   * Scroll to keep the active word visible — only when truly off-screen,
   * with a cooldown to prevent constant jumping.
   */
  function scrollToWord(span) {
    var now = Date.now();
    if (now - lastScrollTime < 5000) return; // 5s cooldown between scrolls
    var rect = span.getBoundingClientRect();
    var viewH = window.innerHeight;
    // Only scroll if the word is fully outside the viewport
    if (rect.bottom < 0 || rect.top > viewH) {
      span.scrollIntoView({ behavior: "smooth", block: "center" });
      lastScrollTime = now;
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
