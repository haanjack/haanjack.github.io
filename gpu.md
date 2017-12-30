---
layout: page
permalink: /gpu/
title: GPU
description: NVIDIA GPU technology & programming에 대한 소개
---

<ul class="post-list">
{% for post in site.gpu reversed %}
    <li>
      <h2><a class="post-title" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h2>
      <p class="post-meta">{{ post.description }}</p>
      <p class="post-meta">{{ post.date | date: "%B %-d, %Y" }}</p>
      <p class="post-meta">{% include post_tags.html %}</p>
      <br/>
      <hr/>
    </li>
{% endfor %}
</ul>
