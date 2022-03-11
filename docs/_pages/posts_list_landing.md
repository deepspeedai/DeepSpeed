---
layout: archive
permalink: /posts-list/
---

{% assign sorted_tags = (site.tags | sort:0) %}
<ul class="tag-box">
	{% for tag in sorted_tags %}
		{% assign t = tag | first %}
		{% assign ps = tag | last %}
		<li><a href="#{{ t | downcase }}">{{ t }} <span class="size">({{ ps.size }})</span></a></li>
	{% endfor %}
</ul>

{% for tag in sorted_tags %}
  {% assign t = tag | first %}
  {% assign posts = tag | last %}
  <div style="text-transform:capitalize;">
    <h4 id="{{ t | downcase }}">{{ t }}</h4>
  </div>
  <ul>
  {% for post in posts %}
    {% if post.tags contains t %}
      {% if post.link %}
        <li>
          <span class="date">{{ post.date | date: '%d %b %y' }}</span>:  <a href="{{ post.link }}">{{ post.title }}</a>
        </li>
      {% else %}
        <li>
          <span class="date">{{ post.date | date: '%d %b %y' }}</span>:  <a href="{{ post.url }}">{{ post.title }}</a>
        </li>
      {% endif %}
    {% endif %}
  {% endfor %}
  </ul>
{% endfor %}
