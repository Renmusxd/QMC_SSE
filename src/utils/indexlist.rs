/// The index list tracks (K,V) pairs with O(1) lookup and O(max(K.into::<usize>())) memory.
/// Returns handles to V entries for quick lookup.
pub struct IndexList<K,V> {
    // May be rearranged with deletions
    indices: Vec<IndexEntry<K>>,
    // K always points to same location
    values: Vec<Option<IndexEntry<V>>>,
}

impl<K,V> Default for IndexList<K,V> {
    fn default() -> Self {
        Self {
            indices: vec![],
            values: vec![]
        }
    }
}

impl<K> IndexList<K,()> where K: Indexable {
    pub fn add(&mut self, index: K) -> IndexListHandle {
        self.place(index, ())
    }
}

impl<K,V> IndexList<K,V> where K: Indexable {
    pub fn place(&mut self, index: K, value: V) -> IndexListHandle {
        let usize_index = index.index();
        if self.values.len() <= usize_index {
            self.values.resize_with(usize_index+1, || None);
        }
        let indices_index = self.indices.len();
        self.indices.push(IndexEntry { index: usize_index, entry: index });
        self.values[usize_index] = Some(IndexEntry { index: indices_index, entry: value });

        IndexListHandle {
            index: usize_index
        }
    }

    pub fn get(&self, index: &IndexListHandle) -> Option<&V> {
        self.values[index.index].as_ref().map(|e| &e.entry)
    }

    pub fn remove(&mut self, index: &IndexListHandle) -> V {
        let entry = self.values[index.index].take().unwrap();
        // To remove from self.indices, swap with end and reorganize
        let index_index = entry.index;
        let last_index = self.indices.len()-1;
        if index_index != last_index {
            let to_edit_entry_index = self.indices[last_index].index;
            self.values[to_edit_entry_index].as_mut().unwrap().index = index_index;
            self.indices.swap(index_index, last_index);
        }
        self.indices.pop();

        entry.entry
    }

    pub fn get_keys(&self) -> impl IntoIterator<Item=&K> {
        self.indices.iter().map(|e| &e.entry)
    }

    pub fn get_keys_and_value(&self) -> impl IntoIterator<Item=(&K, &V)> {
        self.indices.iter().map(|e| {
            let v = &self.values[e.index].as_ref().unwrap().entry;
            (&e.entry, v)
        })
    }

    pub fn clear(&mut self) {
        self.indices.clear();
        self.values.clear();
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<K,V> IndexList<K,V> where K: Indexable + Eq {
    pub fn exists(&self, index: &K) -> bool {
        let index_index = index.index();
        self.values[index_index].as_ref().map(|e| {
            let entry = &self.indices[e.index];
            entry.entry.eq(index)
        }).unwrap_or(false)
    }

    pub fn remove_key(&mut self, index: &K) {
        if self.exists(index) {
            let handle = IndexListHandle { index: index.index() };
            self.remove(&handle);
        }
    }
}

impl<K,V> FromIterator<(K,V)> for IndexList<K,V> where K: Indexable {
    fn from_iter<T: IntoIterator<Item=(K,V)>>(iter: T) -> Self {
        let mut obj = Self::default();
        iter.into_iter().for_each(|(k,v)| {
            obj.place(k,v);
        });
        obj
    }
}

impl<K> FromIterator<K> for IndexList<K,()> where K: Indexable {
    fn from_iter<T: IntoIterator<Item=K>>(iter: T) -> Self {
        iter.into_iter().map(|x| (x, ())).collect()
    }
}

struct IndexEntry<T> {
    index: usize,
    entry: T
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct IndexListHandle {
    index: usize
}

pub trait Indexable {
    fn index(&self) -> usize;
}

impl<T> Indexable for T where T: Copy + Into<usize> {
    fn index(&self) -> usize {
        (*self).into()
    }
}

#[cfg(test)]
mod index_list_tests {
    use super::*;

    #[test]
    fn simple_place_test() {
        let mut obj = IndexList::default();

        let handle = obj.place(0usize, "Hello");
        let value = obj.get(&handle);
        assert_eq!(value, Some(&"Hello"));
        assert!(obj.exists(&0));

        obj.remove(&handle);
        let value = obj.get(&handle);
        assert_eq!(value, None);
    }

    #[test]
    fn simple_place_overwrite_test() {
        let mut obj = IndexList::default();

        let handle = obj.place(0usize, "Hello");
        let handle = obj.place(0usize, "World");
        let value = obj.get(&handle);
        assert_eq!(value, Some(&"World"));
        assert!(obj.exists(&0));

        obj.remove(&handle);
        let value = obj.get(&handle);
        assert_eq!(value, None);
    }

    #[test]
    fn multiple_place_test() {
        let mut obj = IndexList::default();

        let handle_a = obj.place(0usize, "Hello");
        let handle_b = obj.place(1usize, "World");
        let value = obj.get(&handle_a);
        assert_eq!(value, Some(&"Hello"));
        assert!(obj.exists(&0));
        assert!(obj.exists(&1));
        let value = obj.get(&handle_b);
        assert_eq!(value, Some(&"World"));

        obj.remove(&handle_a);
        let value = obj.get(&handle_a);
        assert_eq!(value, None);
        let value = obj.get(&handle_b);
        assert_eq!(value, Some(&"World"));
        assert!(!obj.exists(&0));
        assert!(obj.exists(&1));
    }
}