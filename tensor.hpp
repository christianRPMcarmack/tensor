#pragma once
#include <vector>
#include <numeric>
#include <tuple>
#ifdef __DEBUG__TENSOR__
#include <assert.h>
#define __ASSERT(expression) assert(expression)
#endif

namespace cc {

	template<typename T>
	class ziperator {
		T* p1;
		const T* p2;
	public:
		ziperator(T* _p1, const T* _p2) : p1(_p1), p2(_p2) {}

		std::tuple<T&, const T&> operator*() {
			return std::tuple<T&, const T&>(*p1, *p2);
		}

		void operator++() {
			p1 = p1 + 1;
			p2 = p2 + 1;
		}

		bool operator!=(const ziperator& in) const {
			return (in.p1 != p1 && in.p2 != p2);
		}

		// iterator traits
		using difference_type = T;
		using value_type = T;
		using pointer = T*;
		using reference = T &;
		using iterator_category = std::forward_iterator_tag;
	};

	// zip two vectors for iterating
	template<typename T>
	class zip {
		T* p1begin;
		T* p1end;
		const T* p2begin;
		const T* p2end;
	public:
		zip(std::vector<T>& v1, const std::vector<T>& v2) : p1begin(v1.data()), p1end(v1.data() + v1.size()), p2begin(v2.data()), p2end(v2.data() + v2.size()) {}
		
		ziperator<T> begin() {
			return ziperator<T>(p1begin, p2begin);
		}
		ziperator<T> end() {
			return ziperator<T>(p1end, p2end);
		}
	};

	template<typename T>
	class tensor {
	private:
		/* ---------------- PROPERTIES -------------- */
		std::vector<size_t> mShape; // _shape of array i.e. 2x3 matrix
		std::vector<T> mData; // vector where the data is stored

		/* ------------- PRIVATE METHODS ------------ */
		template <typename accessType>
		T& getRef(std::initializer_list<accessType>&& _indices) {
			std::vector<size_t> indices;
			size_t skip = 0;
			size_t index = 0;
			for (auto& _index : _indices) {
				indices.emplace_back(_index);
				skip = std::accumulate(mShape.begin() + indices.size(), mShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
				index += skip * _index;
			}
			return mData[index];
		}
		// size and resize array given list of dimension lengths
		void sizeArray(const tensor<T>& inArray) {
			for (auto &in : inArray) {
				mData = in.mData;
				mShape = in.mShape;
				break;
			}
		}
		template <typename dimType>
		void sizeArray(std::initializer_list<dimType>&& newShape) {
			if (!mShape.empty()) {
				mShape.clear();
			}
			mShape.reserve(newShape.size());
			for (auto dim : newShape) {
				mShape.emplace_back(dim);
			}
			mData.resize(static_cast<size_t>(std::accumulate(mShape.begin(), mShape.end(), static_cast<size_t>(1), std::multiplies<size_t>())));
		}
	public:
		/* --------------- CONSTRUCTORS ------------- */
		tensor() {}

		template <typename... dimType>
		tensor(dimType ... _shape) { sizeArray({ std::forward<dimType>(_shape)... }); }

		/* --------------- RESIZE TENSOR ------------ */
		template <typename... dimType>
		void resize(dimType&& ... _shape) { sizeArray({ std::forward<dimType>(_shape)... }); }

		/* --------------- ITERATORS ---------------- */
		class iterator {
			T* pData;
			std::vector<size_t> shape;
			std::vector<size_t> indices;
			size_t skip;
			size_t index;
		public:
			// inputs are optional pointer to tensor object and index of current dimension
			iterator(T* _pData, const std::vector<size_t>& _shape = {}, const size_t& _index = 0) : pData(_pData), shape(_shape), indices({ _index }), skip(1), index(0) {
				skip = std::accumulate(shape.begin() + indices.size(), shape.end(), 1, std::multiplies<size_t>());
				index = skip * _index;
			}

			// increments
			iterator& operator++() {
				index += skip;
				return *this;
			}
			iterator operator++(int) {
				iterator retval = *this; ++(*this); return retval;
			}

			// boolean operators
			bool operator==(iterator other) const {
				return pData == other.pData;
			}
			bool operator!=(iterator other) const {
				return !(*this == other);
			}

			// get ref to value
			T& operator*() {
				return *(pData + index);
			}
			operator T& () {
				return *(pData + index);
			}

			// iterate this iterator to the index point
			iterator& operator[](const size_t& _index) {
				indices.emplace_back(_index);
				skip = std::accumulate(shape.begin() + indices.size(), shape.end(), 1, std::multiplies<size_t>());
				index += skip * _index;
				return *this;
			}

			// edit data with input to iterator
			iterator& operator=(const T& in) {
				*(pData + index) = in;
				return *this;
			}

			// iterator traits
			using difference_type = T;
			using value_type = T;
			using pointer = T *;
			using reference = T &;
			using iterator_category = std::forward_iterator_tag;
		};
		auto begin() { return mData.begin(); }
		auto end() { return mData.end(); }
		const auto begin() const { return mData.begin(); }
		const auto end() const { return mData.end(); }

		/* ----------- GET PROPERTIES --------------- */
		const auto& size() const { return mData.size(); }
		const auto& shape() const { return mShape; }

		/* ----------- ACCESS OPERATORS ------------- */
		// [] operator for accessing iterators along dimensions i.e. C++/Python style
		iterator operator[](const size_t& index) {
			__ASSERT(!mShape.empty());
			return iterator(mData.data(), mShape, { index });
		}
		template <typename... accessType>
		T& operator()(accessType&& ... _indices) { return getRef({ std::forward<accessType>(_indices)... }); }

		/* ----------- COPY CONSTRUCTORS ------------ */
		tensor& operator=(const tensor& in) {
			mShape = in.mShape;
			mData = in.mData;
			return *this;
		}

		/* ------- ARITHMETIC OPERATORS ------------- */
		tensor& operator+=(const tensor& in) {
			__ASSERT(in.mShape == this->mShape);
			for (auto [x, y] : zip(this->mData, in.mData)) {
				x += y;
			}
			return *this;
		}
		tensor& operator-=(const tensor& in) {
			__ASSERT(in.mShape == this->mShape);
			for (auto [x, y] : zip(this->mData, in.mData)) {
				x -= y;
			}
			return *this;
		}
		tensor& operator/=(const tensor& in) {
			__ASSERT(in.mShape == this->mShape);
			for (auto [x, y] : zip(this->mData, in.mData)) {
				x /= y;
			}
			return *this;
		}
		tensor& operator*=(const tensor& in) {
			__ASSERT(in.mShape == this->mShape);
			for (auto [x, y] : zip(this->mData, in.mData)) {
				x *= y;
			}
			return *this;
		}

		/* -------- NEW ARITHMETIC OPERATORS -------- */
		tensor operator+(const tensor& in) {
			tensor<T> out = *this;
			__ASSERT(in.mShape == this->mShape);
			for (auto [x, y] : zip(out.mData, in.mData)) {
				x += y;
			}
			return out;
		}
		tensor operator-(const tensor& in) {
			tensor<T> out = *this;
			__ASSERT(in.mShape == this->mShape);
			for (auto [x, y] : zip(out.mData, in.mData)) {
				x -= y;
			}
			return out;
		}
		tensor operator/(const tensor& in) {
			tensor<T> out = *this;
			__ASSERT(in.mShape == this->mShape);
			for (auto [x, y] : zip(out.mData, in.mData)) {
				x /= y;
			}
			return out;
		}
		tensor operator*(const tensor& in) {
			tensor<T> out = *this;
			__ASSERT(in.mShape == this->mShape);
			for (auto [x, y] : zip(out.mData, in.mData)) {
				x *= y;
			}
			return out;
		}
	};
}