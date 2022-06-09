<template>
  <div>
    <div :key="anime" v-for="anime in animes">
      <Anime :anime="anime" />
    </div>
  </div>
</template>

<script>
import Anime from "./Anime.vue";
export default {
  name: "Animes",
  props: {
    animes: Array,
  },
  components: {
    Anime,
  },
};
</script>

<style>
</style>