package org.apache.mahout.cf.taste.impl.common;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class Bicluster<E> implements Comparable<Bicluster<E>> {
	
	private Set<E> users;
	private Set<E> items;
	
	private Bicluster(Set<E> u, Set<E> j) {
		this.users = u;
		this.items = j;
	}
	
	public Bicluster<E> copy() {
		return new Bicluster<E>(new HashSet<E>(this.users), new HashSet<E>(this.items));
	}
	
	public void merge(Bicluster<E> b) {
		Iterator<E> it;
		it = b.getUsers();
		while (it.hasNext()) {
			E x = it.next();
			this.addUser(x);
		}
		it = b.getItems();
		while (it.hasNext()) {
			E x = it.next();
			this.addItem(x);
		}
	}
	
	Bicluster() {
		this.users = new HashSet<E>();
		this.items = new HashSet<E>();
	}
	
	void addUser(E user) {
		this.users.add(user);
	}
	
	void removeUser(E user) {
		this.users.remove(user);
	}
	
	void addItem(E item) {
		this.items.add(item);
	}
	
	void removeItem(E item) {
		this.items.remove(item);
	}
	
	public int getNbUsers() {
		return this.users.size();
	}
	
	public int getNbItems() {
		return this.items.size();
	}
	
	public boolean containsUser(E user) {
		return this.users.contains(user);
	}
	
	public boolean containsItem(E item) {
		return this.items.contains(item);
	}
	
	public Iterator<E> getUsers() {
		return this.users.iterator();
	}
	
	public Iterator<E> getItems() {
		return this.items.iterator();
	}
	
	public boolean isEmpty() {
		return this.users.isEmpty()  || this.items.isEmpty();
	}
	
	public String toString() {
		return this.users.toString() + "x" + this.items.toString();
	}

	@Override
	public int compareTo(Bicluster<E> b) {
		int n = Math.min(this.getNbUsers(), this.getNbItems());
		int m = Math.min(b.getNbUsers(), b.getNbItems());
		if (n < m) {
			return -1;
		} else if (n == m) {
			return 0;
		} else {
			return 1;
		}
	}
	
	public float overlap(Bicluster<E> other) {
		Iterator<E> it;
		int commonU = 0;
		it = this.getUsers();
		while (it.hasNext()) {
			E x = it.next();
			if (other.containsUser(x)) {
				commonU++;
			}
		}
		int commonI = 0;
		it = this.getItems();
		while (it.hasNext()) {
			E x = it.next();
			if (other.containsItem(x)) {
				commonI++;
			}
		}
		return (float) (commonU * commonI) / (float) (this.getNbUsers() * this.getNbItems());
	}

}
